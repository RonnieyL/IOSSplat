import Foundation
import Metal
import simd
import Accelerate

/**
 Converts gaussian data between MetalSplat and OpenSplat formats

 MetalSplat format:
 - Splat { position, color, covA, covB }
 - Covariance stored as 6 values (upper triangle)

 OpenSplat format:
 - Separate buffers: means3d, scales, quats, colors, opacities
 - Covariance decomposed into scale (3) + rotation quaternion (4)
 */
class GaussianConverter {

    private let device: MTLDevice

    /// Temporary CPU-side storage for conversion
    /// In production, this could be done on GPU for better performance
    private struct GaussianCPU {
        var position: SIMD3<Float>
        var scale: SIMD3<Float>
        var quaternion: SIMD4<Float>
        var color: SIMD3<Float>
        var opacity: Float
    }

    init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Public API

    /**
     Convert MetalSplat buffer to OpenSplat format

     - Parameters:
       - splatBuffer: Source buffer with MetalSplat Splat structs
       - count: Number of splats
     - Returns: OpenSplatRenderer.GaussianData with converted buffers
     */
    func convert(splatBuffer: MTLBuffer, count: Int) -> OpenSplatRenderer.GaussianData? {
        // Read splat data from GPU buffer
        let splatPtr = splatBuffer.contents().bindMemory(to: SplatRenderer.Splat.self, capacity: count)

        // Convert to CPU format
        var gaussians: [GaussianCPU] = []
        gaussians.reserveCapacity(count)

        for i in 0..<count {
            let splat = splatPtr[i]
            let gaussian = convertSplat(splat)
            gaussians.append(gaussian)
        }

        // Allocate OpenSplat buffers
        guard let means3dBuffer = device.makeBuffer(length: count * 3 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let scalesBuffer = device.makeBuffer(length: count * 3 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let quatsBuffer = device.makeBuffer(length: count * 4 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let colorsBuffer = device.makeBuffer(length: count * 3 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let opacitiesBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            return nil
        }

        // Write converted data to buffers (packed format for OpenSplat)
        let means3dPtr = means3dBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)
        let scalesPtr = scalesBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)
        let quatsPtr = quatsBuffer.contents().bindMemory(to: Float.self, capacity: count * 4)
        let colorsPtr = colorsBuffer.contents().bindMemory(to: Float.self, capacity: count * 3)
        let opacitiesPtr = opacitiesBuffer.contents().bindMemory(to: Float.self, capacity: count)

        for i in 0..<count {
            let g = gaussians[i]

            // Packed float3 (no padding)
            means3dPtr[i * 3 + 0] = g.position.x
            means3dPtr[i * 3 + 1] = g.position.y
            means3dPtr[i * 3 + 2] = g.position.z

            scalesPtr[i * 3 + 0] = g.scale.x
            scalesPtr[i * 3 + 1] = g.scale.y
            scalesPtr[i * 3 + 2] = g.scale.z

            // Packed float4
            quatsPtr[i * 4 + 0] = g.quaternion.x
            quatsPtr[i * 4 + 1] = g.quaternion.y
            quatsPtr[i * 4 + 2] = g.quaternion.z
            quatsPtr[i * 4 + 3] = g.quaternion.w

            colorsPtr[i * 3 + 0] = g.color.x
            colorsPtr[i * 3 + 1] = g.color.y
            colorsPtr[i * 3 + 2] = g.color.z

            opacitiesPtr[i] = g.opacity
        }

        return OpenSplatRenderer.GaussianData(
            means3d: means3dBuffer,
            scales: scalesBuffer,
            quats: quatsBuffer,
            colors: colorsBuffer,
            opacities: opacitiesBuffer,
            count: count
        )
    }

    // MARK: - Private Conversion

    private func convertSplat(_ splat: SplatRenderer.Splat) -> GaussianCPU {
        // Extract position
        let position = SIMD3<Float>(splat.position.x, splat.position.y, splat.position.z)

        // Extract color and opacity
        let color = SIMD3<Float>(
            Float(splat.color.r),
            Float(splat.color.g),
            Float(splat.color.b)
        )
        let opacity = Float(splat.color.a)

        // Decompose covariance into scale + rotation
        let (scale, quaternion) = decomposeCovarianceToScaleRotation(
            covA: splat.covA,
            covB: splat.covB
        )

        return GaussianCPU(
            position: position,
            scale: scale,
            quaternion: quaternion,
            color: color,
            opacity: opacity
        )
    }

    /**
     Decompose 3x3 covariance matrix into scale + rotation

     Covariance matrix (symmetric):
     [covA.x  covA.y  covA.z]
     [covA.y  covB.x  covB.y]
     [covA.z  covB.y  covB.z]

     Decomposition: Σ = R * S * S^T * R^T
     where S is diagonal scale matrix, R is rotation matrix

     Approach: Eigenvalue decomposition
     - Eigenvalues → scales (sqrt of eigenvalues)
     - Eigenvectors → rotation matrix → quaternion
     */
    private func decomposeCovarianceToScaleRotation(
        covA: SplatRenderer.PackedHalf3,
        covB: SplatRenderer.PackedHalf3
    ) -> (scale: SIMD3<Float>, quaternion: SIMD4<Float>) {
        // Build symmetric 3x3 covariance matrix
        var covMatrix: [Float] = [
            Float(covA.x), Float(covA.y), Float(covA.z),  // Row 0
            Float(covA.y), Float(covB.x), Float(covB.y),  // Row 1
            Float(covA.z), Float(covB.y), Float(covB.z)   // Row 2
        ]

        // Eigenvalue decomposition using Accelerate
        // This gives us eigenvalues (scales^2) and eigenvectors (rotation)
        var N = __CLPK_integer(3)
        var N_lda = __CLPK_integer(3)  // Leading dimension
        var jobz = Int8(86)  // 'V' - compute eigenvalues and eigenvectors
        var uplo = Int8(85)  // 'U' - upper triangle
        var lwork = __CLPK_integer(9)
        var work = [Float](repeating: 0, count: 9)
        var eigenvalues = [Float](repeating: 0, count: 3)
        var info = __CLPK_integer(0)

        // Call LAPACK's ssyev for symmetric eigenvalue decomposition
        ssyev_(&jobz, &uplo, &N, &covMatrix, &N_lda, &eigenvalues, &work, &lwork, &info)

        if info != 0 {
            // Decomposition failed, return identity rotation and default scale
            print("⚠️ Eigenvalue decomposition failed, using identity")
            return (
                scale: SIMD3<Float>(0.1, 0.1, 0.1),
                quaternion: SIMD4<Float>(1, 0, 0, 0)  // Identity quaternion (w=1, x=y=z=0)
            )
        }

        // Extract scales (sqrt of eigenvalues, ensure positive)
        let scale = SIMD3<Float>(
            sqrt(max(eigenvalues[0], 1e-6)),
            sqrt(max(eigenvalues[1], 1e-6)),
            sqrt(max(eigenvalues[2], 1e-6))
        )

        // Extract rotation matrix from eigenvectors (column-major)
        let rotationMatrix = simd_float3x3(
            SIMD3<Float>(covMatrix[0], covMatrix[1], covMatrix[2]),  // Column 0
            SIMD3<Float>(covMatrix[3], covMatrix[4], covMatrix[5]),  // Column 1
            SIMD3<Float>(covMatrix[6], covMatrix[7], covMatrix[8])   // Column 2
        )

        // Convert rotation matrix to quaternion
        let quaternion = matrixToQuaternion(rotationMatrix)

        return (scale: scale, quaternion: quaternion)
    }

    /**
     Convert 3x3 rotation matrix to quaternion

     Uses Shepperd's method (numerically stable)
     */
    private func matrixToQuaternion(_ m: simd_float3x3) -> SIMD4<Float> {
        // Compute trace
        let trace = m[0][0] + m[1][1] + m[2][2]

        var quat: SIMD4<Float>

        if trace > 0 {
            // w is largest component
            let s = sqrt(trace + 1.0) * 2  // s = 4 * w
            quat = SIMD4<Float>(
                0.25 * s,                           // w
                (m[2][1] - m[1][2]) / s,           // x
                (m[0][2] - m[2][0]) / s,           // y
                (m[1][0] - m[0][1]) / s            // z
            )
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            // x is largest component
            let s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2  // s = 4 * x
            quat = SIMD4<Float>(
                (m[2][1] - m[1][2]) / s,           // w
                0.25 * s,                           // x
                (m[0][1] + m[1][0]) / s,           // y
                (m[0][2] + m[2][0]) / s            // z
            )
        } else if m[1][1] > m[2][2] {
            // y is largest component
            let s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2  // s = 4 * y
            quat = SIMD4<Float>(
                (m[0][2] - m[2][0]) / s,           // w
                (m[0][1] + m[1][0]) / s,           // x
                0.25 * s,                           // y
                (m[1][2] + m[2][1]) / s            // z
            )
        } else {
            // z is largest component
            let s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2  // s = 4 * z
            quat = SIMD4<Float>(
                (m[1][0] - m[0][1]) / s,           // w
                (m[0][2] + m[2][0]) / s,           // x
                (m[1][2] + m[2][1]) / s,           // y
                0.25 * s                            // z
            )
        }

        // Normalize quaternion
        let length = sqrt(quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w)
        if length > 1e-6 {
            quat /= length
        }

        // Note: OpenSplat expects quaternion as (x, y, z, w), adjust if needed
        // Current output is (w, x, y, z) based on SIMD4 ordering
        return SIMD4<Float>(quat.y, quat.z, quat.w, quat.x)  // Reorder to (x, y, z, w)
    }
}
