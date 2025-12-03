import Metal
import MetalKit
import simd

/// GaussianSplatting renderer using the OpenSplat Metal rasterizer pipeline.
///
/// ## Pipeline Overview
/// The forward pass consists of these stages:
/// 1. **Project Gaussians** - Transform 3D Gaussians to 2D screen space
/// 2. **Bin & Sort** - Assign Gaussians to tiles and sort by depth
/// 3. **Rasterize** - Render sorted Gaussians per tile with alpha blending
///
/// ## Forward Pass Inputs
/// - `means3d`: Float3 buffer - 3D positions of Gaussian centers
/// - `scales`: Float3 buffer - Scale in each axis (use raw scale, not log-space)
/// - `quats`: Float4 buffer - Rotation quaternions (w, x, y, z)
/// - `colors`: Float3 buffer - RGB colors (or SH coefficients for view-dependent color)
/// - `opacities`: Float buffer - Opacity values [0, 1]
/// - `viewMatrix`: 4x4 matrix - Camera view matrix (world to camera)
/// - `projMatrix`: 4x4 matrix - Camera projection matrix
/// - `intrinsics`: (fx, fy, cx, cy) - Camera intrinsics
/// - `imageSize`: (width, height) - Output image dimensions
///
/// ## Forward Pass Outputs
/// - `outImage`: Float3 buffer - Rendered RGB image (H x W x 3)
/// - `finalTs`: Float buffer - Final transmittance per pixel (for backward pass)
/// - `finalIdx`: Int32 buffer - Last contributing Gaussian per pixel (for backward pass)
///
/// ## Backward Pass (for training)
/// Given gradients on output image, computes gradients for:
/// - dL/d_xy: Gradient w.r.t. projected 2D positions
/// - dL/d_conic: Gradient w.r.t. 2D covariance inverse
/// - dL/d_colors: Gradient w.r.t. colors
/// - dL/d_opacity: Gradient w.r.t. opacities
/// Then project_gaussians_backward computes:
/// - dL/d_means3d, dL/d_scales, dL/d_quats
///
class GaussianSplatting {
    
    // MARK: - Constants
    
    /// Tile size for rasterization (must match shader BLOCK_X, BLOCK_Y)
    static let blockX = 16
    static let blockY = 16
    
    // MARK: - Metal Resources
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // MARK: - Compute Pipeline States
    
    /// Projects 3D Gaussians to 2D screen space
    /// Outputs: cov3d, xys, depths, radii, conics, numTilesHit
    private let projectGaussiansPipeline: MTLComputePipelineState
    
    /// Maps each Gaussian to the tiles it overlaps
    /// Outputs: isectIds (tile|depth key), gaussianIds
    private let mapGaussianToIntersectsPipeline: MTLComputePipelineState
    
    /// Finds start/end indices for each tile in sorted intersection list
    /// Outputs: tileBins (start, end) per tile
    private let getTileBinEdgesPipeline: MTLComputePipelineState
    
    /// Rasterizes Gaussians with N-channel colors (e.g., RGB)
    /// Outputs: outImg, finalTs, finalIdx
    private let ndRasterizeForwardPipeline: MTLComputePipelineState
    
    /// Copies buffer to texture for display
    private let displayPipeline: MTLComputePipelineState
    
    /// Backward pass for rasterization (computes dL/d_xy, dL/d_conic, dL/d_colors, dL/d_opacity)
    private let ndRasterizeBackwardPipeline: MTLComputePipelineState?
    
    /// Backward pass for projection (computes dL/d_means, dL/d_scales, dL/d_quats)
    private let projectGaussiansBackwardPipeline: MTLComputePipelineState?
    
    /// Computes spherical harmonics for view-dependent colors
    private let computeSHForwardPipeline: MTLComputePipelineState?
    
    // MARK: - Render Pipeline for Display
    
    private var copyPipelineState: MTLRenderPipelineState?
    private var intermediateTexture: MTLTexture?
    
    // MARK: - Gaussian Data Buffers (CPU-writable)
    
    /// 3D positions of Gaussian centers - Float3 packed
    private var means3dBuffer: MTLBuffer!
    /// Scales - Float3 packed
    private var scalesBuffer: MTLBuffer!
    /// Rotation quaternions (w,x,y,z) - Float4 packed
    private var quatsBuffer: MTLBuffer!
    /// RGB colors - Float3 packed
    private var colorsBuffer: MTLBuffer!
    /// Opacity values - Float
    private var opacitiesBuffer: MTLBuffer!
    
    // MARK: - Intermediate Buffers (GPU-only)
    
    /// Upper-triangular 3D covariance (6 floats per Gaussian)
    private var cov3dBuffer: MTLBuffer!
    /// Projected 2D screen positions - Float2 packed
    private var xysBuffer: MTLBuffer!
    /// Depth values (camera Z)
    private var depthsBuffer: MTLBuffer!
    /// Screen-space radius per Gaussian
    private var radiiBuffer: MTLBuffer!
    /// 2D covariance inverse (conic) - Float3 packed
    private var conicsBuffer: MTLBuffer!
    /// Number of tiles hit per Gaussian
    private var numTilesHitBuffer: MTLBuffer!
    /// Cumulative tiles hit (prefix sum result)
    private var cumTilesHitBuffer: MTLBuffer!
    
    // MARK: - Sorting Buffers
    
    /// Intersection IDs: (tileId << 32 | depthBits) for sorting
    private var isectIdsBuffer: MTLBuffer!
    /// Gaussian IDs corresponding to each intersection
    private var gaussianIdsBuffer: MTLBuffer!
    /// Sorted Gaussian IDs
    private var gaussianIdsSortedBuffer: MTLBuffer!
    /// Tile bin ranges: (start, end) per tile - Int2 packed
    private var tileBinsBuffer: MTLBuffer!
    
    // MARK: - Output Buffers
    
    /// Rendered image - Float3 packed (H x W x 3)
    private var outImageBuffer: MTLBuffer!
    /// Final transmittance per pixel
    private var finalTsBuffer: MTLBuffer!
    /// Last contributing Gaussian index per pixel
    private var finalIdxBuffer: MTLBuffer!
    
    // MARK: - State
    
    private let maxPoints: Int
    private var pointCount: Int = 0
    private var maxIntersects: Int
    private var frameCount: Int = 0
    
    // MARK: - Initialization
    
    init?(device: MTLDevice, commandQueue: MTLCommandQueue, library: MTLLibrary, maxPoints: Int = 500_000) {
        self.device = device
        self.commandQueue = commandQueue
        self.library = library
        self.maxPoints = maxPoints
        self.maxIntersects = maxPoints * 64 // Estimate: each Gaussian hits ~64 tiles on average
        
        // Create compute pipelines
        do {
            projectGaussiansPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "project_gaussians_forward_kernel")!)
            mapGaussianToIntersectsPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "map_gaussian_to_intersects_kernel")!)
            getTileBinEdgesPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "get_tile_bin_edges_kernel")!)
            ndRasterizeForwardPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "nd_rasterize_forward_kernel")!)
            displayPipeline = try device.makeComputePipelineState(
                function: library.makeFunction(name: "display_texture_kernel")!)
            
            // Optional backward pipelines (for training)
            if let fn = library.makeFunction(name: "nd_rasterize_backward_kernel") {
                ndRasterizeBackwardPipeline = try device.makeComputePipelineState(function: fn)
            } else {
                ndRasterizeBackwardPipeline = nil
            }
            if let fn = library.makeFunction(name: "project_gaussians_backward_kernel") {
                projectGaussiansBackwardPipeline = try device.makeComputePipelineState(function: fn)
            } else {
                projectGaussiansBackwardPipeline = nil
            }
            if let fn = library.makeFunction(name: "compute_sh_forward_kernel") {
                computeSHForwardPipeline = try device.makeComputePipelineState(function: fn)
            } else {
                computeSHForwardPipeline = nil
            }
        } catch {
            print("[GaussianSplatting] Failed to create compute pipeline: \(error)")
            return nil
        }
        
        // Allocate Gaussian data buffers (CPU-writable for dynamic updates)
        allocateGaussianBuffers()
        
        // Allocate intermediate buffers (GPU-only)
        allocateIntermediateBuffers()
    }
    
    private func allocateGaussianBuffers() {
        let float3Stride = MemoryLayout<Float>.stride * 3
        let float4Stride = MemoryLayout<Float>.stride * 4
        
        // Use storageModeShared for CPU write access
        means3dBuffer = device.makeBuffer(length: maxPoints * float3Stride, options: .storageModeShared)
        scalesBuffer = device.makeBuffer(length: maxPoints * float3Stride, options: .storageModeShared)
        quatsBuffer = device.makeBuffer(length: maxPoints * float4Stride, options: .storageModeShared)
        colorsBuffer = device.makeBuffer(length: maxPoints * float3Stride, options: .storageModeShared)
        opacitiesBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Float>.stride, options: .storageModeShared)
    }
    
    private func allocateIntermediateBuffers() {
        // Projection outputs - use storageModeShared for CPU access during sorting
        cov3dBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Float>.stride * 6, options: .storageModePrivate)
        xysBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Float>.stride * 2, options: .storageModeShared)
        depthsBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Float>.stride, options: .storageModeShared)
        radiiBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Int32>.stride, options: .storageModeShared)
        conicsBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Float>.stride * 3, options: .storageModeShared)
        numTilesHitBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Int32>.stride, options: .storageModeShared)
        cumTilesHitBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Int32>.stride, options: .storageModeShared)
        
        // Sorting buffers - storageModeShared for CPU sort
        isectIdsBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int64>.stride, options: .storageModeShared)
        gaussianIdsBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int32>.stride, options: .storageModeShared)
        gaussianIdsSortedBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int32>.stride, options: .storageModeShared)
    }
    
    // MARK: - Public API
    
    /// Current number of Gaussians
    var count: Int { pointCount }
    
    /// Add new Gaussians with positions, colors, and optional parameters
    func addPoints(positions: [SIMD3<Float>], 
                   colors: [SIMD3<Float>],
                   scales: [SIMD3<Float>]? = nil,
                   rotations: [simd_quatf]? = nil,
                   opacities: [Float]? = nil) {
        
        guard positions.count == colors.count else {
            print("[GaussianSplatting] Position/color count mismatch")
            return
        }
        
        let count = positions.count
        if pointCount + count > maxPoints {
            print("[GaussianSplatting] Max points reached, cannot add \(count) points")
            return
        }
        
        // Write means (positions)
        let meansPtr = means3dBuffer.contents()
            .advanced(by: pointCount * MemoryLayout<Float>.stride * 3)
            .assumingMemoryBound(to: Float.self)
        for (i, pos) in positions.enumerated() {
            meansPtr[i * 3 + 0] = pos.x
            meansPtr[i * 3 + 1] = pos.y
            meansPtr[i * 3 + 2] = pos.z
        }
        
        // Write colors
        let colorsPtr = colorsBuffer.contents()
            .advanced(by: pointCount * MemoryLayout<Float>.stride * 3)
            .assumingMemoryBound(to: Float.self)
        for (i, col) in colors.enumerated() {
            colorsPtr[i * 3 + 0] = col.x
            colorsPtr[i * 3 + 1] = col.y
            colorsPtr[i * 3 + 2] = col.z
        }
        
        // Write scales (default: very small spheres - 2mm radius)
        let scalesPtr = scalesBuffer.contents()
            .advanced(by: pointCount * MemoryLayout<Float>.stride * 3)
            .assumingMemoryBound(to: Float.self)
        let defaultScale: Float = 0.01 // 2mm radius - small enough to avoid huge tile coverage
        for i in 0..<count {
            if let s = scales?[i] {
                scalesPtr[i * 3 + 0] = s.x
                scalesPtr[i * 3 + 1] = s.y
                scalesPtr[i * 3 + 2] = s.z
            } else {
                scalesPtr[i * 3 + 0] = defaultScale
                scalesPtr[i * 3 + 1] = defaultScale
                scalesPtr[i * 3 + 2] = defaultScale
            }
        }
        
        // Write quaternions (default: identity)
        let quatsPtr = quatsBuffer.contents()
            .advanced(by: pointCount * MemoryLayout<Float>.stride * 4)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            if let q = rotations?[i] {
                quatsPtr[i * 4 + 0] = q.real        // w
                quatsPtr[i * 4 + 1] = q.imag.x     // x
                quatsPtr[i * 4 + 2] = q.imag.y     // y
                quatsPtr[i * 4 + 3] = q.imag.z     // z
            } else {
                quatsPtr[i * 4 + 0] = 1.0  // w (identity)
                quatsPtr[i * 4 + 1] = 0.0  // x
                quatsPtr[i * 4 + 2] = 0.0  // y
                quatsPtr[i * 4 + 3] = 0.0  // z
            }
        }
        
        // Write opacities (default: semi-transparent to avoid saturation)
        let opacitiesPtr = opacitiesBuffer.contents()
            .advanced(by: pointCount * MemoryLayout<Float>.stride)
            .assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            // Default to 0.5 opacity for better blending when close to Gaussians
            opacitiesPtr[i] = opacities?[i] ?? 0.5
        }
        
        pointCount += count
        
        // Debug: print sample of added colors
        print("[GaussianSplatting] Added \(count) points. Sample colors: ")
        for i in 0..<min(3, count) {
            print("  Point \(i): color=(\(colors[i].x), \(colors[i].y), \(colors[i].z))")
        }
    }
    
    /// Clear all Gaussians
    func clear() {
        pointCount = 0
    }
    
    // MARK: - Forward Pass
    
    /// Render Gaussians to the output texture
    ///
    /// - Parameters:
    ///   - cameraTransform: Camera-to-world transform (from ARKit camera.transform)
    ///   - projectionMatrix: Camera projection matrix (column-major)
    ///   - viewportSize: Output image size
    ///   - outputTexture: Texture to render to
    ///   - drawable: Optional drawable for presentation
    func draw(cameraTransform: matrix_float4x4,
              projectionMatrix: matrix_float4x4,
              viewportSize: CGSize,
              outputTexture: MTLTexture,
              drawable: CAMetalDrawable?) {
        
        guard pointCount > 0 else {
            print("[GaussianSplatting] No points to render")
            return
        }
        
        // Limit points to render to avoid GPU hang
        // TODO: Implement proper GPU sorting for more points
        let maxRenderPoints = 5000
        let renderCount = min(pointCount, maxRenderPoints)
        
        let width = Int(viewportSize.width)
        let height = Int(viewportSize.height)
        let tileWidth = (width + Self.blockX - 1) / Self.blockX
        let tileHeight = (height + Self.blockY - 1) / Self.blockY
        let numTiles = tileWidth * tileHeight
        
        if pointCount > maxRenderPoints {
            print("[GaussianSplatting] Rendering \(renderCount)/\(pointCount) Gaussians (limited) to \(width)x\(height)")
        } else {
            print("[GaussianSplatting] Rendering \(renderCount) Gaussians to \(width)x\(height)")
        }
        
        // Ensure tile bins buffer is large enough
        let tileBinsSize = numTiles * MemoryLayout<Int32>.stride * 2
        if tileBinsBuffer == nil || tileBinsBuffer.length < tileBinsSize {
            tileBinsBuffer = device.makeBuffer(length: tileBinsSize, options: .storageModeShared)
        }
        
        // Ensure output buffers are large enough
        let pixelCount = width * height
        if outImageBuffer == nil || outImageBuffer.length < pixelCount * 3 * MemoryLayout<Float>.stride {
            outImageBuffer = device.makeBuffer(length: pixelCount * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)
            finalTsBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.stride, options: .storageModePrivate)
            finalIdxBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        }
        
        // ============================================
        // STAGE 1: Project Gaussians
        // ============================================
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        computeEncoder.setComputePipelineState(projectGaussiansPipeline)
        
        // Kernel arguments (must match shader signature)
        var numPoints = Int32(renderCount)
        computeEncoder.setBytes(&numPoints, length: MemoryLayout<Int32>.size, index: 0)
        computeEncoder.setBuffer(means3dBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(scalesBuffer, offset: 0, index: 2)
        var globScale: Float = 1.0
        computeEncoder.setBytes(&globScale, length: MemoryLayout<Float>.size, index: 3)
        computeEncoder.setBuffer(quatsBuffer, offset: 0, index: 4)
        
        // ===========================================
        // MATCHING OPENSPLAT EXACTLY
        // ===========================================
        // 
        // OpenSplat (model.cpp lines 85-115):
        // 1. Gets fx, fy, cx, cy from camera intrinsics (focal length in pixels)
        // 2. Gets camToWorld transform, extracts R and T
        // 3. Flips Y and Z: R = R * diag(1, -1, -1)
        // 4. Computes viewMat (worldToCam) = inverse of camToWorld
        // 5. Computes fovX = 2 * atan(width / (2 * fx)), fovY = 2 * atan(height / (2 * fy))
        // 6. Builds custom OpenGL projection matrix using fov
        // 7. Passes projMat * viewMat to shader
        //
        // ARKit gives us:
        // - cameraTransform: camera-to-world transform (like OpenSplat's camToWorld)
        // - projectionMatrix: Metal projection matrix
        //
        // We need to:
        // 1. Extract R and T from cameraTransform
        // 2. Apply flip: R = R * diag(1, -1, -1)
        // 3. Invert: Rinv = R.T, Tinv = -Rinv * T
        // 4. Build viewMat = [Rinv | Tinv]
        
        // Extract focal lengths from ARKit projection matrix
        let fx = abs(projectionMatrix.columns.0.x) * Float(width) / 2.0
        let fy = abs(projectionMatrix.columns.1.y) * Float(height) / 2.0
        let cx = Float(width) / 2.0
        let cy = Float(height) / 2.0
        
        // Compute FOV like OpenSplat does
        let fovX = 2.0 * atan(Float(width) / (2.0 * fx))
        let fovY = 2.0 * atan(Float(height) / (2.0 * fy))
        
        // Build OpenGL-style projection matrix exactly like OpenSplat
        let zNear: Float = 0.001
        let zFar: Float = 1000.0
        let t = zNear * tan(0.5 * fovY)
        let b = -t
        let r = zNear * tan(0.5 * fovX)
        let l = -r
        
        // OpenSplat's projection matrix (from model.cpp lines 35-47):
        // Row-major as stored in PyTorch:
        //     {2n/(r-l),    0,          (r+l)/(r-l),  0},           <- row 0
        //     {0,           2n/(t-b),   (t+b)/(t-b),  0},           <- row 1  
        //     {0,           0,          (f+n)/(f-n),  -fn/(f-n)},   <- row 2
        //     {0,           0,          1,            0}            <- row 3
        //
        // Swift uses column-major, so we build columns:
        let gsplatProjMatrix = matrix_float4x4(columns: (
            SIMD4<Float>(2*zNear/(r-l), 0, 0, 0),                    // column 0
            SIMD4<Float>(0, 2*zNear/(t-b), 0, 0),                    // column 1
            SIMD4<Float>((r+l)/(r-l), (t+b)/(t-b), (zFar+zNear)/(zFar-zNear), 1),  // column 2
            SIMD4<Float>(0, 0, -zFar*zNear/(zFar-zNear), 0)          // column 3
        ))
        
        // ============================================
        // Build View Matrix exactly like OpenSplat (model.cpp lines 92-109)
        // ============================================
        // Step 1: Extract R (rotation) and T (translation) from cameraTransform
        // cameraTransform is camera-to-world, format: [R | T] with last row [0,0,0,1]
        // In column-major (Swift): columns 0-2 are rotation, column 3 is translation
        
        // Extract 3x3 rotation (upper-left 3x3)
        var R = matrix_float3x3(
            SIMD3<Float>(cameraTransform.columns.0.x, cameraTransform.columns.0.y, cameraTransform.columns.0.z),
            SIMD3<Float>(cameraTransform.columns.1.x, cameraTransform.columns.1.y, cameraTransform.columns.1.z),
            SIMD3<Float>(cameraTransform.columns.2.x, cameraTransform.columns.2.y, cameraTransform.columns.2.z)
        )
        
        // Extract translation (column 3)
        let T = SIMD3<Float>(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
        
        // Step 2: Flip Y and Z: R = R * diag(1, -1, -1)
        // This converts from ARKit (Y-up, -Z forward) to gsplat (Y-down, +Z forward)
        let flipYZMat = matrix_float3x3(
            SIMD3<Float>(1, 0, 0),
            SIMD3<Float>(0, -1, 0),
            SIMD3<Float>(0, 0, -1)
        )
        R = R * flipYZMat
        
        // Step 3: Invert to get world-to-camera
        // Rinv = R^T (transpose of orthonormal rotation)
        let Rinv = R.transpose
        
        // Tinv = -Rinv * T
        let Tinv = -(Rinv * T)
        
        // Step 4: Build 4x4 view matrix [Rinv | Tinv]
        // Column-major layout
        let viewMatrix = matrix_float4x4(columns: (
            SIMD4<Float>(Rinv.columns.0.x, Rinv.columns.0.y, Rinv.columns.0.z, 0),
            SIMD4<Float>(Rinv.columns.1.x, Rinv.columns.1.y, Rinv.columns.1.z, 0),
            SIMD4<Float>(Rinv.columns.2.x, Rinv.columns.2.y, Rinv.columns.2.z, 0),
            SIMD4<Float>(Tinv.x, Tinv.y, Tinv.z, 1)
        ))
        
        // OpenSplat passes projMat * viewMat to the shader
        let combinedMatrix = gsplatProjMatrix * viewMatrix
        
        // Matrices: OpenSplat shaders expect ROW-MAJOR, Swift uses column-major
        // Transpose to convert column-major to row-major
        var viewMat = viewMatrix.transpose
        var projMat = combinedMatrix.transpose
        
        // Debug: Print key values
        if frameCount % 60 == 0 {  // Print every 60 frames
            print("[GaussianSplatting] DEBUG:")
            print("  fx=\(fx), fy=\(fy), cx=\(cx), cy=\(cy)")
            print("  fovX=\(fovX * 180/Float.pi)°, fovY=\(fovY * 180/Float.pi)°")
            print("  gsplatProj diag: [\(gsplatProjMatrix.columns.0.x), \(gsplatProjMatrix.columns.1.y), \(gsplatProjMatrix.columns.2.z), \(gsplatProjMatrix.columns.3.w)]")
            print("  Camera position (T): [\(T.x), \(T.y), \(T.z)]")
            print("  View matrix translation (Tinv): [\(Tinv.x), \(Tinv.y), \(Tinv.z)]")
        }
        frameCount += 1
        
        computeEncoder.setBytes(&viewMat, length: MemoryLayout<matrix_float4x4>.size, index: 5)
        computeEncoder.setBytes(&projMat, length: MemoryLayout<matrix_float4x4>.size, index: 6)
        
        // Camera intrinsics for covariance projection
        var intrins = SIMD4<Float>(fx, fy, cx, cy)
        computeEncoder.setBytes(&intrins, length: MemoryLayout<SIMD4<Float>>.size, index: 7)
        
        var imgSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        computeEncoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 8)
        
        var tileBounds = SIMD3<UInt32>(UInt32(tileWidth), UInt32(tileHeight), 1)
        computeEncoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 9)
        
        var clipThresh: Float = 0.01
        computeEncoder.setBytes(&clipThresh, length: MemoryLayout<Float>.size, index: 10)
        
        // Output buffers
        computeEncoder.setBuffer(cov3dBuffer, offset: 0, index: 11)
        computeEncoder.setBuffer(xysBuffer, offset: 0, index: 12)
        computeEncoder.setBuffer(depthsBuffer, offset: 0, index: 13)
        computeEncoder.setBuffer(radiiBuffer, offset: 0, index: 14)
        computeEncoder.setBuffer(conicsBuffer, offset: 0, index: 15)
        computeEncoder.setBuffer(numTilesHitBuffer, offset: 0, index: 16)
        
        let projectGridSize = MTLSize(width: renderCount, height: 1, depth: 1)
        let projectGroupSize = MTLSize(width: min(renderCount, projectGaussiansPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        computeEncoder.dispatchThreads(projectGridSize, threadsPerThreadgroup: projectGroupSize)
        computeEncoder.endEncoding()
        
        // Wait for projection to complete (need numTilesHit for CPU prefix sum)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // ============================================
        // DEBUG: Check projection results
        // ============================================
        let radiiPtr = radiiBuffer.contents().bindMemory(to: Int32.self, capacity: renderCount)
        let depthsPtr = depthsBuffer.contents().bindMemory(to: Float.self, capacity: renderCount)
        let xysPtr = xysBuffer.contents().bindMemory(to: Float.self, capacity: renderCount * 2)
        let meansPtr = means3dBuffer.contents().bindMemory(to: Float.self, capacity: renderCount * 3)
        
        var validCount = 0
        var minX: Float = Float.greatestFiniteMagnitude
        var maxX: Float = -Float.greatestFiniteMagnitude
        var minY: Float = Float.greatestFiniteMagnitude
        var maxY: Float = -Float.greatestFiniteMagnitude
        
        // Also track world positions of valid points
        var sampleWorldPos: [(Float, Float, Float)] = []
        
        for i in 0..<min(renderCount, 100) {
            if radiiPtr[i] > 0 {
                validCount += 1
                let x = xysPtr[i * 2]
                let y = xysPtr[i * 2 + 1]
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
                
                if sampleWorldPos.count < 3 {
                    let wx = meansPtr[i * 3]
                    let wy = meansPtr[i * 3 + 1]
                    let wz = meansPtr[i * 3 + 2]
                    sampleWorldPos.append((wx, wy, wz))
                }
            }
        }
        
        if validCount > 0 {
            print("[GaussianSplatting] XY range: x=[\(Int(minX)),\(Int(maxX))], y=[\(Int(minY)),\(Int(maxY))], viewport=\(width)x\(height)")
            if !sampleWorldPos.isEmpty {
                print("[GaussianSplatting] Sample world positions: \(sampleWorldPos)")
            }
        }
        
        // ============================================
        // STAGE 2: Prefix Sum (CPU) to get cumulative tile counts
        // ============================================
        let numTilesHitPtr = numTilesHitBuffer.contents().bindMemory(to: Int32.self, capacity: renderCount)
        let cumTilesHitPtr = cumTilesHitBuffer.contents().bindMemory(to: Int32.self, capacity: renderCount)
        
        var totalIntersects = 0
        for i in 0..<renderCount {
            cumTilesHitPtr[i] = Int32(totalIntersects)
            totalIntersects += Int(numTilesHitPtr[i])
        }
        
        // Cap total intersects to prevent GPU hang
        let maxSafeIntersects = 500_000 // Safe limit for mobile GPU
        if totalIntersects > maxSafeIntersects {
            print("[GaussianSplatting] WARNING: Intersects \(totalIntersects) > \(maxSafeIntersects), skipping frame")
            return
        }
        
        print("[GaussianSplatting] Total intersects: \(totalIntersects), valid first 100: \(validCount)")
        
        guard totalIntersects > 0 else {
            print("[GaussianSplatting] No visible Gaussians")
            return
        }
        
        // Ensure intersection buffers are large enough
        if totalIntersects > maxIntersects {
            print("[GaussianSplatting] Expanding intersection buffers: \(totalIntersects) > \(maxIntersects)")
            maxIntersects = totalIntersects * 2
            isectIdsBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int64>.stride, options: .storageModeShared)
            gaussianIdsBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int32>.stride, options: .storageModeShared)
            gaussianIdsSortedBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int32>.stride, options: .storageModeShared)
        }
        
        // ============================================
        // STAGE 3: Map Gaussians to Tile Intersections
        // ============================================
        guard let mapCommandBuffer = commandQueue.makeCommandBuffer(),
              let mapEncoder = mapCommandBuffer.makeComputeCommandEncoder() else { return }
        
        mapEncoder.setComputePipelineState(mapGaussianToIntersectsPipeline)
        mapEncoder.setBytes(&numPoints, length: MemoryLayout<Int32>.size, index: 0)
        mapEncoder.setBuffer(xysBuffer, offset: 0, index: 1)
        mapEncoder.setBuffer(depthsBuffer, offset: 0, index: 2)
        mapEncoder.setBuffer(radiiBuffer, offset: 0, index: 3)
        mapEncoder.setBuffer(cumTilesHitBuffer, offset: 0, index: 4)  // Use cumulative, not numTilesHit
        mapEncoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 5)
        mapEncoder.setBuffer(isectIdsBuffer, offset: 0, index: 6)
        mapEncoder.setBuffer(gaussianIdsBuffer, offset: 0, index: 7)
        
        mapEncoder.dispatchThreads(projectGridSize, threadsPerThreadgroup: projectGroupSize)
        mapEncoder.endEncoding()
        
        mapCommandBuffer.commit()
        mapCommandBuffer.waitUntilCompleted()
        
        // ============================================
        // STAGE 4: Sort Intersections by (tileId, depth)
        // ============================================
        // CPU sort (GPU radix sort could be added for better performance)
        let isectIdsPtr = isectIdsBuffer.contents().bindMemory(to: Int64.self, capacity: totalIntersects)
        let gaussianIdsPtr = gaussianIdsBuffer.contents().bindMemory(to: Int32.self, capacity: totalIntersects)
        let gaussianIdsSortedPtr = gaussianIdsSortedBuffer.contents().bindMemory(to: Int32.self, capacity: totalIntersects)
        
        // Create index array and sort by isectIds
        var indices = Array(0..<totalIntersects)
        indices.sort { isectIdsPtr[$0] < isectIdsPtr[$1] }
        
        // Also sort isectIds in-place for tile bin edges
        var sortedIsectIds = [Int64](repeating: 0, count: totalIntersects)
        for (newIdx, oldIdx) in indices.enumerated() {
            gaussianIdsSortedPtr[newIdx] = gaussianIdsPtr[oldIdx]
            sortedIsectIds[newIdx] = isectIdsPtr[oldIdx]
        }
        memcpy(isectIdsPtr, sortedIsectIds, totalIntersects * MemoryLayout<Int64>.stride)
        
        // ============================================
        // STAGE 5: Get Tile Bin Edges
        // ============================================
        // Zero out tile bins
        memset(tileBinsBuffer.contents(), 0, tileBinsSize)
        
        guard let binCommandBuffer = commandQueue.makeCommandBuffer(),
              let binEncoder = binCommandBuffer.makeComputeCommandEncoder() else { return }
        
        binEncoder.setComputePipelineState(getTileBinEdgesPipeline)
        var numIntersects = Int32(totalIntersects)
        binEncoder.setBytes(&numIntersects, length: MemoryLayout<Int32>.size, index: 0)
        binEncoder.setBuffer(isectIdsBuffer, offset: 0, index: 1)
        binEncoder.setBuffer(tileBinsBuffer, offset: 0, index: 2)
        
        let binGridSize = MTLSize(width: totalIntersects, height: 1, depth: 1)
        let binGroupSize = MTLSize(width: min(totalIntersects, getTileBinEdgesPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        binEncoder.dispatchThreads(binGridSize, threadsPerThreadgroup: binGroupSize)
        binEncoder.endEncoding()
        
        binCommandBuffer.commit()
        binCommandBuffer.waitUntilCompleted()
        
        // ============================================
        // STAGE 6: Rasterize
        // ============================================
        
        // CRITICAL: Zero the output image buffer before rasterization!
        // The rasterization kernel accumulates colors (+=), so we must clear it each frame
        // otherwise colors saturate to white over time.
        let outImageSize = pixelCount * 3 * MemoryLayout<Float>.stride
        memset(outImageBuffer.contents(), 0, outImageSize)
        
        guard let rasterCommandBuffer = commandQueue.makeCommandBuffer(),
              let rasterEncoder = rasterCommandBuffer.makeComputeCommandEncoder() else { return }
        
        rasterEncoder.setComputePipelineState(ndRasterizeForwardPipeline)
        
        rasterEncoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 0)
        var imgSize3 = SIMD3<UInt32>(UInt32(width), UInt32(height), 1)
        rasterEncoder.setBytes(&imgSize3, length: MemoryLayout<SIMD3<UInt32>>.size, index: 1)
        var channels: UInt32 = 3
        rasterEncoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 2)
        rasterEncoder.setBuffer(gaussianIdsSortedBuffer, offset: 0, index: 3)
        rasterEncoder.setBuffer(tileBinsBuffer, offset: 0, index: 4)
        rasterEncoder.setBuffer(xysBuffer, offset: 0, index: 5)
        rasterEncoder.setBuffer(conicsBuffer, offset: 0, index: 6)
        rasterEncoder.setBuffer(colorsBuffer, offset: 0, index: 7)
        rasterEncoder.setBuffer(opacitiesBuffer, offset: 0, index: 8)
        rasterEncoder.setBuffer(finalTsBuffer, offset: 0, index: 9)
        rasterEncoder.setBuffer(finalIdxBuffer, offset: 0, index: 10)
        rasterEncoder.setBuffer(outImageBuffer, offset: 0, index: 11)
        
        var background = SIMD3<Float>(0, 0, 0)
        rasterEncoder.setBytes(&background, length: MemoryLayout<SIMD3<Float>>.size, index: 12)
        
        var blockDim = SIMD2<UInt32>(UInt32(Self.blockX), UInt32(Self.blockY))
        rasterEncoder.setBytes(&blockDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 13)
        
        let rasterGridSize = MTLSize(width: width, height: height, depth: 1)
        let rasterGroupSize = MTLSize(width: Self.blockX, height: Self.blockY, depth: 1)
        rasterEncoder.dispatchThreads(rasterGridSize, threadsPerThreadgroup: rasterGroupSize)
        rasterEncoder.endEncoding()
        
        // ============================================
        // STAGE 7: Copy to Output Texture
        // ============================================
        // Create/resize intermediate texture if needed
        if intermediateTexture == nil ||
           intermediateTexture!.width != width ||
           intermediateTexture!.height != height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: outputTexture.pixelFormat,
                width: width, height: height, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            intermediateTexture = device.makeTexture(descriptor: desc)
            
            // Create copy pipeline if needed
            if copyPipelineState == nil {
                if let vf = library.makeFunction(name: "copyVertex"),
                   let ff = library.makeFunction(name: "copyFragment") {
                    let pd = MTLRenderPipelineDescriptor()
                    pd.vertexFunction = vf
                    pd.fragmentFunction = ff
                    pd.colorAttachments[0].pixelFormat = outputTexture.pixelFormat
                    copyPipelineState = try? device.makeRenderPipelineState(descriptor: pd)
                }
            }
        }
        
        guard let intermediateTex = intermediateTexture else { return }
        
        // Copy buffer to intermediate texture
        guard let displayEncoder = rasterCommandBuffer.makeComputeCommandEncoder() else { return }
        displayEncoder.setComputePipelineState(displayPipeline)
        displayEncoder.setBuffer(outImageBuffer, offset: 0, index: 0)
        displayEncoder.setTexture(intermediateTex, index: 0)
        
        let displayGridSize = MTLSize(width: width, height: height, depth: 1)
        let displayGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        displayEncoder.dispatchThreads(displayGridSize, threadsPerThreadgroup: displayGroupSize)
        displayEncoder.endEncoding()
        
        // Copy intermediate texture to output drawable
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = outputTexture
        rpd.colorAttachments[0].loadAction = .dontCare
        rpd.colorAttachments[0].storeAction = .store
        
        if let copyPipeline = copyPipelineState,
           let renderEncoder = rasterCommandBuffer.makeRenderCommandEncoder(descriptor: rpd) {
            renderEncoder.setRenderPipelineState(copyPipeline)
            renderEncoder.setFragmentTexture(intermediateTex, index: 0)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renderEncoder.endEncoding()
        }
        
        // Present drawable if provided
        if let drawable = drawable {
            rasterCommandBuffer.present(drawable)
        }
        
        rasterCommandBuffer.commit()
    }
    
    // MARK: - Backward Pass (for training)
    
    /// Compute gradients for training
    /// Call after forward pass with gradient of loss w.r.t. output image
    ///
    /// The backward pass computes:
    /// 1. `nd_rasterize_backward_kernel`: dL/d_xy, dL/d_conic, dL/d_colors, dL/d_opacity
    /// 2. `project_gaussians_backward_kernel`: dL/d_means, dL/d_scales, dL/d_quats
    func backward(gradOutput: MTLBuffer,
                  viewMatrix: matrix_float4x4,
                  projectionMatrix: matrix_float4x4,
                  viewportSize: CGSize) -> (
                      gradMeans: MTLBuffer,
                      gradScales: MTLBuffer,
                      gradQuats: MTLBuffer,
                      gradColors: MTLBuffer,
                      gradOpacities: MTLBuffer
                  )? {
        
        guard let rasterBackward = ndRasterizeBackwardPipeline,
              let projectBackward = projectGaussiansBackwardPipeline else {
            print("[GaussianSplatting] Backward pipelines not available")
            return nil
        }
        
        // TODO: Implement backward pass
        // 1. Call nd_rasterize_backward_kernel to get dL/d_xy, dL/d_conic, dL/d_colors, dL/d_opacity
        // 2. Call project_gaussians_backward_kernel to get dL/d_means, dL/d_scales, dL/d_quats
        
        return nil
    }
}
