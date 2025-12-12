//
//  OpenSplatRenderer.swift
//  SceneDepthPointCloud
//
//  Pure Swift wrapper for OpenSplat Metal kernels (without PyTorch dependencies)
//

import Foundation
import Metal
import MetalKit
import simd

enum OpenSplatError: Error {
    case metalDeviceNotAvailable
    case metalLibraryLoadFailed(String)
    case kernelFunctionNotFound(String)
    case pipelineStateCreationFailed(String)
    case bufferCreationFailed(String)
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
}

/// OpenSplat renderer using tile-based rasterization
/// Based on the architecture from gsplat_metal.mm but without PyTorch dependencies
class OpenSplatRenderer {

    // MARK: - Metal Resources
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // MARK: - Compute Pipeline States (10 kernels from gsplat_metal.mm)
    private let ndRasterizeForwardPipeline: MTLComputePipelineState
    private let ndRasterizeBackwardPipeline: MTLComputePipelineState
    private let rasterizeBackwardPipeline: MTLComputePipelineState
    private let projectGaussiansForwardPipeline: MTLComputePipelineState
    private let projectGaussiansBackwardPipeline: MTLComputePipelineState
    private let computeSHForwardPipeline: MTLComputePipelineState
    private let computeSHBackwardPipeline: MTLComputePipelineState
    private let computeCov2dBoundsPipeline: MTLComputePipelineState
    private let mapGaussianToIntersectsPipeline: MTLComputePipelineState
    private let getTileBinEdgesPipeline: MTLComputePipelineState

    // MARK: - Constants
    private let blockX: Int = 16
    private let blockY: Int = 16

    // MARK: - Initialization

    init(device: MTLDevice) throws {
        print("🔧 Initializing OpenSplatRenderer...")

        self.device = device

        // Create command queue
        guard let queue = device.makeCommandQueue() else {
            throw OpenSplatError.metalDeviceNotAvailable
        }
        self.commandQueue = queue
        print("   ✓ Command queue created")

        // Load Metal library
        guard let library = device.makeDefaultLibrary() else {
            throw OpenSplatError.metalLibraryLoadFailed("Default library not found")
        }
        self.library = library
        print("   ✓ Metal library loaded with \(library.functionNames.count) functions")

        // Create compute pipeline states for all kernels
        do {
            self.ndRasterizeForwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "nd_rasterize_forward_kernel"
            )
            print("   ✓ nd_rasterize_forward_kernel pipeline created")

            self.ndRasterizeBackwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "nd_rasterize_backward_kernel"
            )
            print("   ✓ nd_rasterize_backward_kernel pipeline created")

            self.rasterizeBackwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "rasterize_backward_kernel"
            )
            print("   ✓ rasterize_backward_kernel pipeline created")

            self.projectGaussiansForwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "project_gaussians_forward_kernel_OPENSPLAT"
            )
            print("   ✓ project_gaussians_forward_kernel_OPENSPLAT pipeline created")

            self.projectGaussiansBackwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "project_gaussians_backward_kernel"
            )
            print("   ✓ project_gaussians_backward_kernel pipeline created")

            self.computeSHForwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "compute_sh_forward_kernel"
            )
            print("   ✓ compute_sh_forward_kernel pipeline created")

            self.computeSHBackwardPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "compute_sh_backward_kernel"
            )
            print("   ✓ compute_sh_backward_kernel pipeline created")

            self.computeCov2dBoundsPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "compute_cov2d_bounds_kernel"
            )
            print("   ✓ compute_cov2d_bounds_kernel pipeline created")

            self.mapGaussianToIntersectsPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "map_gaussian_to_intersects_kernel"
            )
            print("   ✓ map_gaussian_to_intersects_kernel pipeline created")

            self.getTileBinEdgesPipeline = try Self.createPipelineState(
                device: device, library: library, functionName: "get_tile_bin_edges_kernel"
            )
            print("   ✓ get_tile_bin_edges_kernel pipeline created")

        } catch {
            throw error
        }

        print("✅ OpenSplatRenderer initialized successfully with 10 kernels")
    }

    // MARK: - Pipeline State Creation

    private static func createPipelineState(
        device: MTLDevice,
        library: MTLLibrary,
        functionName: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw OpenSplatError.kernelFunctionNotFound(functionName)
        }

        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw OpenSplatError.pipelineStateCreationFailed("\(functionName): \(error.localizedDescription)")
        }
    }

    // MARK: - Rendering Pipeline

    /// Main rendering function - implements the 3-stage OpenSplat pipeline
    func render(
        gaussianMeans: MTLBuffer,      // [N, 3] float
        gaussianScales: MTLBuffer,     // [N, 3] float
        gaussianQuats: MTLBuffer,      // [N, 4] float (quaternions)
        gaussianColors: MTLBuffer,     // [N, 3] float (RGB)
        gaussianOpacities: MTLBuffer,  // [N] float
        numGaussians: Int,
        viewMatrix: simd_float4x4,
        projMatrix: simd_float4x4,
        fx: Float, fy: Float,
        cx: Float, cy: Float,
        imgWidth: Int,
        imgHeight: Int,
        outputTexture: MTLTexture,
        globalScale: Float = 1.0,
        clipThreshold: Float = 0.01,
        background: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    ) throws {

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw OpenSplatError.commandBufferCreationFailed
        }

        // Calculate tile bounds
        let tilesX = (imgWidth + blockX - 1) / blockX
        let tilesY = (imgHeight + blockY - 1) / blockY
        let tileBounds = SIMD4<UInt32>(UInt32(tilesX), UInt32(tilesY), 1, 0xDEAD)

        // STAGE 1: Project Gaussians (3D → 2D)
        let stage1Outputs = try projectGaussiansForward(
            commandBuffer: commandBuffer,
            means3d: gaussianMeans,
            scales: gaussianScales,
            quats: gaussianQuats,
            numPoints: numGaussians,
            viewMatrix: viewMatrix,
            projMatrix: projMatrix,
            fx: fx, fy: fy, cx: cx, cy: cy,
            imgWidth: imgWidth,
            imgHeight: imgHeight,
            tileBounds: tileBounds,
            globalScale: globalScale,
            clipThreshold: clipThreshold
        )

        // Commit and wait to read back debug info
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read back debug counters
        let countersPtr = stage1Outputs.debugCounters.contents().bindMemory(to: Int32.self, capacity: 5)
        let counter0 = countersPtr[0]
        let counter1 = countersPtr[1]
        let counter2 = countersPtr[2]
        let zeroTiles = countersPtr[3]
        let success = countersPtr[4]

        print("   � DEBUG MAGIC NUMBERS:")
        print("      • counter[0] = \(counter0) (expect 77777=started, 99999=processing, or count)")
        print("      • counter[1] = \(counter1) (expect num_points value)")
        print("      • counter[2] = \(counter2) (expect 88888 if thread0 exited early)")
        
        if counter0 == 77777 {
            print("      ✅ Kernel STARTED but didn't process! (thread 0 wrote 77777)")
            print("      📍 num_points received by GPU: \(counter1)")
            if counter2 == 88888 {
                print("      ⚠️ Thread 0 EXITED EARLY at idx >= num_points check!")
            }
        } else if counter0 == 99999 {
            print("      ✅ Kernel PROCESSING! (thread 0 wrote 99999)")
        } else {
            print("      ℹ️ Processed count: \(counter0)")
        }
        
        print("   📊 PROJECTION FAILURE BREAKDOWN:")
        print("      • Zero tile area: \(zeroTiles)")
        print("      • ✅ SUCCESS: \(success)")

        // Read back projection results for debugging
        let radiiPtr = stage1Outputs.radii.contents().bindMemory(to: Int32.self, capacity: numGaussians)
        let depthsPtr = stage1Outputs.depths.contents().bindMemory(to: Float.self, capacity: numGaussians)
        let numTilesHitPtr = stage1Outputs.numTilesHit.contents().bindMemory(to: Int32.self, capacity: numGaussians)

        // Debug: Find ALL successful Gaussians
        var successIndices: [Int] = []
        for i in 0..<numGaussians {
            if radiiPtr[i] > 0 {
                successIndices.append(i)
                if successIndices.count >= 10 { break }
            }
        }

        if !successIndices.isEmpty {
            let firstIdx = successIndices[0]
            print("   🔍 DEBUG - Found successful Gaussians at indices: \(successIndices)")
            print("   🔍 DEBUG - First SUCCESS (index \(firstIdx)):")
            let cov3dPtr = stage1Outputs.cov3d.contents().bindMemory(to: Float.self, capacity: numGaussians * 6)
            let cov3dOffset = firstIdx * 6
            print("      • Radius: \(radiiPtr[firstIdx]) px, Depth: \(String(format: "%.3f", depthsPtr[firstIdx])), Tiles: \(numTilesHitPtr[firstIdx])")
            print("      • Cov3D: [\(String(format: "%.6f", cov3dPtr[cov3dOffset+0]))  \(String(format: "%.6f", cov3dPtr[cov3dOffset+1]))  \(String(format: "%.6f", cov3dPtr[cov3dOffset+2]))]")
            print("               [\(String(format: "%.6f", cov3dPtr[cov3dOffset+1]))  \(String(format: "%.6f", cov3dPtr[cov3dOffset+3]))  \(String(format: "%.6f", cov3dPtr[cov3dOffset+4]))]")
            print("               [\(String(format: "%.6f", cov3dPtr[cov3dOffset+2]))  \(String(format: "%.6f", cov3dPtr[cov3dOffset+4]))  \(String(format: "%.6f", cov3dPtr[cov3dOffset+5]))]")

            // Read debug values written by any successful Gaussian
            if firstIdx == 2357 || firstIdx == 2513 || firstIdx == 2561 || firstIdx < 10 {
                let debugOffset = numGaussians * 6
                print("   🔍 COVARIANCE CALCULATION DEBUG (Gaussian \(firstIdx)):")
                print("      • scale: (\(String(format: "%.6f", cov3dPtr[debugOffset+0])), \(String(format: "%.6f", cov3dPtr[debugOffset+1])), \(String(format: "%.6f", cov3dPtr[debugOffset+2])))")
                print("      • quat: (\(String(format: "%.6f", cov3dPtr[debugOffset+3])), \(String(format: "%.6f", cov3dPtr[debugOffset+4])), \(String(format: "%.6f", cov3dPtr[debugOffset+5])), \(String(format: "%.6f", cov3dPtr[debugOffset+6])))")
                print("      • S diag: (\(String(format: "%.6f", cov3dPtr[debugOffset+7])), \(String(format: "%.6f", cov3dPtr[debugOffset+8])), \(String(format: "%.6f", cov3dPtr[debugOffset+9])))")
                print("      • R diag: (\(String(format: "%.6f", cov3dPtr[debugOffset+10])), \(String(format: "%.6f", cov3dPtr[debugOffset+11])), \(String(format: "%.6f", cov3dPtr[debugOffset+12])))")
                print("      • M diag: (\(String(format: "%.6f", cov3dPtr[debugOffset+13])), \(String(format: "%.6f", cov3dPtr[debugOffset+14])), \(String(format: "%.6f", cov3dPtr[debugOffset+15])))")
                print("      • tmp diag: (\(String(format: "%.6f", cov3dPtr[debugOffset+16])), \(String(format: "%.6f", cov3dPtr[debugOffset+17])), \(String(format: "%.6f", cov3dPtr[debugOffset+18])))")
            }
            print("   🔍 First Gaussian (idx 0) cov3d [magic numbers 888/999?]: [\(String(format: "%.1f", cov3dPtr[0])), \(String(format: "%.1f", cov3dPtr[1])), \(String(format: "%.1f", cov3dPtr[2])), \(String(format: "%.1f", cov3dPtr[3])), \(String(format: "%.1f", cov3dPtr[4])), \(String(format: "%.1f", cov3dPtr[5]))]")
        } else {
            print("   ⚠️ No successful Gaussians found!")
        }

        print("   🔍 DEBUG - First 5 Gaussian projection results:")
        for i in 0..<min(5, numGaussians) {
            let r = radiiPtr[i]
            let d = depthsPtr[i]
            let tiles = numTilesHitPtr[i]
            print("      [\(i)] radius=\(r), depth=\(String(format: "%.3f", d)), tiles=\(tiles)")
        }
        
        // Count how many have non-zero radii
        var totalNonZero = 0
        for i in 0..<numGaussians {
            if radiiPtr[i] > 0 { totalNonZero += 1 }
        }
        print("   🔍 Gaussians with non-zero radius: \(totalNonZero)/\(numGaussians)")
        
        // Count total intersections
        let numIntersects = try countIntersects(numTilesHit: stage1Outputs.numTilesHit, numPoints: numGaussians)

        print("   🔍 Counting intersections from \(numGaussians) Gaussians...")
        print("   🔍 Total intersections found: \(numIntersects)")

        if numIntersects == 0 {
            print("   ⚠️ No Gaussian intersections, skipping render")
            print("   🔍 This means ALL Gaussians failed projection (clipped, out of view, or zero radius)")
            return
        }
        
        // Create new command buffer for remaining stages
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw OpenSplatError.commandBufferCreationFailed
        }

        // STAGE 2: Map Gaussians to tiles and sort
        let stage2Outputs = try mapAndSortGaussians(
            commandBuffer: commandBuffer,
            xys: stage1Outputs.xys,
            depths: stage1Outputs.depths,
            radii: stage1Outputs.radii,
            numTilesHit: stage1Outputs.numTilesHit,
            numPoints: numGaussians,
            numIntersects: numIntersects,
            tileBounds: tileBounds
        )

        // STAGE 3: Rasterize to output texture
        try rasterizeForward(
            commandBuffer: commandBuffer,
            gaussianIdsSorted: stage2Outputs.gaussianIdsSorted,
            tileBins: stage2Outputs.tileBins,
            xys: stage1Outputs.xys,
            conics: stage1Outputs.conics,
            colors: gaussianColors,
            opacities: gaussianOpacities,
            imgWidth: imgWidth,
            imgHeight: imgHeight,
            tileBounds: tileBounds,
            outputTexture: outputTexture,
            background: background
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    // MARK: - Stage 1: Project Gaussians

    private struct ProjectGaussiansOutputs {
        let cov3d: MTLBuffer       // [N, 6] triangular 3D covariance
        let xys: MTLBuffer         // [N, 2] 2D projected positions
        let depths: MTLBuffer      // [N] depth values
        let radii: MTLBuffer       // [N] radii in pixels
        let conics: MTLBuffer      // [N, 3] 2D conics (inverse covariance)
        let numTilesHit: MTLBuffer // [N] number of tiles hit by each Gaussian
        let debugCounters: MTLBuffer // [5] debug counters for projection failures
        let debugPipeline: MTLBuffer // [32] detailed pipeline debug for first Gaussian
    }

    private func projectGaussiansForward(
        commandBuffer: MTLCommandBuffer,
        means3d: MTLBuffer,
        scales: MTLBuffer,
        quats: MTLBuffer,
        numPoints: Int,
        viewMatrix: simd_float4x4,
        projMatrix: simd_float4x4,
        fx: Float, fy: Float,
        cx: Float, cy: Float,
        imgWidth: Int,
        imgHeight: Int,
        tileBounds: SIMD4<UInt32>,
        globalScale: Float,
        clipThreshold: Float
    ) throws -> ProjectGaussiansOutputs {

        // Allocate output buffers (cov3d has extra space for debug data)
        let cov3dSize = (numPoints * 6 + 32) * MemoryLayout<Float>.stride  // +32 floats for debug
        let xysSize = numPoints * 2 * MemoryLayout<Float>.stride
        let depthsSize = numPoints * MemoryLayout<Float>.stride
        let radiiSize = numPoints * MemoryLayout<Int32>.stride
        let conicsSize = numPoints * 3 * MemoryLayout<Float>.stride
        let numTilesHitSize = numPoints * MemoryLayout<Int32>.stride

        // Debug counters: [0]=processed, [1]=clipped, [2]=bad_cov, [3]=zero_tiles, [4]=success
        let debugCountersSize = 5 * MemoryLayout<Int32>.stride
        guard let debugCounters = device.makeBuffer(length: debugCountersSize, options: .storageModeShared) else {
            throw OpenSplatError.bufferCreationFailed("debug counters")
        }
        // Zero out counters
        memset(debugCounters.contents(), 0, debugCountersSize)

        // Debug pipeline buffer for first Gaussian:
        // [0-2]=world_pos, [3-5]=view_pos, [6-11]=cov3d, [12-14]=cov2d,
        // [15]=det, [16]=radius, [17-19]=scale, [20-23]=quat, [24-27]=viewmat_col0
        let debugPipelineSize = 32 * MemoryLayout<Float>.stride
        guard let debugPipeline = device.makeBuffer(length: debugPipelineSize, options: .storageModeShared) else {
            throw OpenSplatError.bufferCreationFailed("debug pipeline")
        }
        memset(debugPipeline.contents(), 0, debugPipelineSize)

        // Use .storageModeShared for debugging (allows CPU readback)
        // TODO: Change back to .storageModePrivate for performance after debugging
        guard let cov3d = device.makeBuffer(length: cov3dSize, options: .storageModeShared),
              let xys = device.makeBuffer(length: xysSize, options: .storageModeShared),
              let depths = device.makeBuffer(length: depthsSize, options: .storageModeShared),
              let radii = device.makeBuffer(length: radiiSize, options: .storageModeShared),
              let conics = device.makeBuffer(length: conicsSize, options: .storageModeShared),
              let numTilesHit = device.makeBuffer(length: numTilesHitSize, options: .storageModeShared) else {
            throw OpenSplatError.bufferCreationFailed("project_gaussians_forward outputs")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw OpenSplatError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(projectGaussiansForwardPipeline)

        // Encode arguments (matches gsplat_metal.mm lines 381-399)
        var numPointsVar = Int32(numPoints)
        var globScaleVar = globalScale
        var intrinsics = SIMD4<Float>(fx, fy, cx, cy)
        var imgSize = SIMD2<UInt32>(UInt32(imgWidth), UInt32(imgHeight))
        var tileBoundsVar = tileBounds
        var clipThreshVar = clipThreshold
        // After testing: DO NOT transpose! 
        // The kernel reconstruction already handles column-major properly
        // Original matrix gives geometrically correct view space Z values
        var viewMatVar = viewMatrix
        var projMatVar = projMatrix

        encoder.setBytes(&numPointsVar, length: MemoryLayout<Int32>.stride, index: 0)
        encoder.setBuffer(means3d, offset: 0, index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        encoder.setBytes(&globScaleVar, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setBuffer(quats, offset: 0, index: 4)
        encoder.setBytes(&viewMatVar, length: MemoryLayout<simd_float4x4>.stride, index: 5)
        encoder.setBytes(&projMatVar, length: MemoryLayout<simd_float4x4>.stride, index: 6)
        encoder.setBytes(&intrinsics, length: MemoryLayout<SIMD4<Float>>.stride, index: 7)
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 8)
        encoder.setBytes(&tileBoundsVar, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 9)
        encoder.setBytes(&clipThreshVar, length: MemoryLayout<Float>.stride, index: 10)
        encoder.setBuffer(cov3d, offset: 0, index: 11)
        encoder.setBuffer(xys, offset: 0, index: 12)
        encoder.setBuffer(depths, offset: 0, index: 13)
        encoder.setBuffer(radii, offset: 0, index: 14)
        encoder.setBuffer(conics, offset: 0, index: 15)
        encoder.setBuffer(numTilesHit, offset: 0, index: 16)
        encoder.setBuffer(debugCounters, offset: 0, index: 17)
        encoder.setBuffer(debugPipeline, offset: 0, index: 18)

        // Dispatch
        let threadsPerThreadgroup = min(
            projectGaussiansForwardPipeline.maxTotalThreadsPerThreadgroup,
            numPoints
        )
        let threadgroups = MTLSize(
            width: (numPoints + threadsPerThreadgroup - 1) / threadsPerThreadgroup,
            height: 1,
            depth: 1
        )
        let threadsPerGrid = MTLSize(width: numPoints, height: 1, depth: 1)

        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: MTLSize(
            width: threadsPerThreadgroup, height: 1, depth: 1
        ))
        encoder.endEncoding()

        print("   🔍 Projection kernel dispatched with:")
        print("      • clip_thresh=\(clipThreshold)")
        print("      • global_scale=\(globalScale)")
        print("      • tile_bounds=(\(tileBounds.x), \(tileBounds.y))")
        print("      • intrinsics: fx=\(fx), fy=\(fy), cx=\(cx), cy=\(cy)")
        print("      • img_size: \(imgWidth) × \(imgHeight)")
        
        // Compute what OpenSplat will calculate
        let tanFovX = 0.5 * Float(imgWidth) / fx
        let tanFovY = 0.5 * Float(imgHeight) / fy
        print("      • tan_fov computed: x=\(tanFovX), y=\(tanFovY)")

        return ProjectGaussiansOutputs(
            cov3d: cov3d,
            xys: xys,
            depths: depths,
            radii: radii,
            conics: conics,
            numTilesHit: numTilesHit,
            debugCounters: debugCounters,
            debugPipeline: debugPipeline
        )
    }

    // MARK: - Stage 2: Map and Sort

    private struct MapAndSortOutputs {
        let gaussianIdsSorted: MTLBuffer  // [num_intersects] sorted Gaussian indices
        let tileBins: MTLBuffer           // [num_intersects, 2] tile bin edges
    }

    private func mapAndSortGaussians(
        commandBuffer: MTLCommandBuffer,
        xys: MTLBuffer,
        depths: MTLBuffer,
        radii: MTLBuffer,
        numTilesHit: MTLBuffer,
        numPoints: Int,
        numIntersects: Int,
        tileBounds: SIMD4<UInt32>
    ) throws -> MapAndSortOutputs {

        // Allocate buffers
        let gaussianIdsSize = numIntersects * MemoryLayout<Int32>.stride
        let isectIdsSize = numIntersects * 2 * MemoryLayout<UInt32>.stride // Packed as 2×uint32

        guard let gaussianIdsUnsorted = device.makeBuffer(length: gaussianIdsSize, options: .storageModePrivate),
              let isectIdsUnsorted = device.makeBuffer(length: isectIdsSize, options: .storageModePrivate),
              let gaussianIdsSorted = device.makeBuffer(length: gaussianIdsSize, options: .storageModePrivate),
              let isectIdsSorted = device.makeBuffer(length: isectIdsSize, options: .storageModePrivate) else {
            throw OpenSplatError.bufferCreationFailed("map_gaussian_to_intersects buffers")
        }

        // Step 1: Map Gaussians to intersects
        guard let encoder1 = commandBuffer.makeComputeCommandEncoder() else {
            throw OpenSplatError.commandEncoderCreationFailed
        }

        encoder1.setComputePipelineState(mapGaussianToIntersectsPipeline)

        var numPointsVar = Int32(numPoints)
        var tileBoundsVar = tileBounds

        encoder1.setBytes(&numPointsVar, length: MemoryLayout<Int32>.stride, index: 0)
        encoder1.setBuffer(xys, offset: 0, index: 1)
        encoder1.setBuffer(depths, offset: 0, index: 2)
        encoder1.setBuffer(radii, offset: 0, index: 3)
        encoder1.setBuffer(numTilesHit, offset: 0, index: 4)
        encoder1.setBytes(&tileBoundsVar, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 5)
        encoder1.setBuffer(isectIdsUnsorted, offset: 0, index: 6)
        encoder1.setBuffer(gaussianIdsUnsorted, offset: 0, index: 7)

        let threadsPerThreadgroup1 = min(mapGaussianToIntersectsPipeline.maxTotalThreadsPerThreadgroup, numPoints)
        encoder1.dispatchThreads(
            MTLSize(width: numPoints, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup1, height: 1, depth: 1)
        )
        encoder1.endEncoding()

        // Step 2: Sort by isect_ids (depth + tile_id)
        // TODO: Implement proper radix sort here
        // For now, just copy unsorted → sorted (this will cause rendering artifacts)
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw OpenSplatError.commandEncoderCreationFailed
        }
        blitEncoder.copy(from: isectIdsUnsorted, sourceOffset: 0, to: isectIdsSorted, destinationOffset: 0, size: isectIdsSize)
        blitEncoder.copy(from: gaussianIdsUnsorted, sourceOffset: 0, to: gaussianIdsSorted, destinationOffset: 0, size: gaussianIdsSize)
        blitEncoder.endEncoding()

        // Step 3: Get tile bin edges
        let tileBinsSize = numIntersects * 2 * MemoryLayout<Int32>.stride
        guard let tileBins = device.makeBuffer(length: tileBinsSize, options: .storageModePrivate) else {
            throw OpenSplatError.bufferCreationFailed("tile_bins buffer")
        }

        guard let encoder3 = commandBuffer.makeComputeCommandEncoder() else {
            throw OpenSplatError.commandEncoderCreationFailed
        }

        encoder3.setComputePipelineState(getTileBinEdgesPipeline)

        var numIntersectsVar = Int32(numIntersects)
        encoder3.setBytes(&numIntersectsVar, length: MemoryLayout<Int32>.stride, index: 0)
        encoder3.setBuffer(isectIdsSorted, offset: 0, index: 1)
        encoder3.setBuffer(tileBins, offset: 0, index: 2)

        let threadsPerThreadgroup3 = min(getTileBinEdgesPipeline.maxTotalThreadsPerThreadgroup, numIntersects)
        encoder3.dispatchThreads(
            MTLSize(width: numIntersects, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup3, height: 1, depth: 1)
        )
        encoder3.endEncoding()

        return MapAndSortOutputs(
            gaussianIdsSorted: gaussianIdsSorted,
            tileBins: tileBins
        )
    }

    // MARK: - Stage 3: Rasterize

    private func rasterizeForward(
        commandBuffer: MTLCommandBuffer,
        gaussianIdsSorted: MTLBuffer,
        tileBins: MTLBuffer,
        xys: MTLBuffer,
        conics: MTLBuffer,
        colors: MTLBuffer,
        opacities: MTLBuffer,
        imgWidth: Int,
        imgHeight: Int,
        tileBounds: SIMD4<UInt32>,
        outputTexture: MTLTexture,
        background: SIMD3<Float>
    ) throws {

        // Create temporary output buffer (will convert to texture after)
        let channels: UInt32 = 3
        let outImgSize = imgHeight * imgWidth * Int(channels) * MemoryLayout<Float>.stride
        let finalTsSize = imgHeight * imgWidth * MemoryLayout<Float>.stride
        let finalIdxSize = imgHeight * imgWidth * MemoryLayout<Int32>.stride

        guard let outImg = device.makeBuffer(length: outImgSize, options: .storageModePrivate),
              let finalTs = device.makeBuffer(length: finalTsSize, options: .storageModePrivate),
              let finalIdx = device.makeBuffer(length: finalIdxSize, options: .storageModePrivate) else {
            throw OpenSplatError.bufferCreationFailed("rasterize_forward outputs")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw OpenSplatError.commandEncoderCreationFailed
        }

        encoder.setComputePipelineState(ndRasterizeForwardPipeline)

        var tileBoundsVar = tileBounds
        var imgSize = SIMD4<UInt32>(UInt32(imgWidth), UInt32(imgHeight), 1, 0xDEAD)
        var channelsVar = channels
        var backgroundVar = background
        var blockSize = SIMD2<Int32>(Int32(blockX), Int32(blockY))

        encoder.setBytes(&tileBoundsVar, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 0)
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 1)
        encoder.setBytes(&channelsVar, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBuffer(gaussianIdsSorted, offset: 0, index: 3)
        encoder.setBuffer(tileBins, offset: 0, index: 4)
        encoder.setBuffer(xys, offset: 0, index: 5)
        encoder.setBuffer(conics, offset: 0, index: 6)
        encoder.setBuffer(colors, offset: 0, index: 7)
        encoder.setBuffer(opacities, offset: 0, index: 8)
        encoder.setBuffer(finalTs, offset: 0, index: 9)
        encoder.setBuffer(finalIdx, offset: 0, index: 10)
        encoder.setBuffer(outImg, offset: 0, index: 11)
        encoder.setBytes(&backgroundVar, length: MemoryLayout<SIMD3<Float>>.stride, index: 12)
        encoder.setBytes(&blockSize, length: MemoryLayout<SIMD2<Int32>>.stride, index: 13)

        // Dispatch using tile-based threading
        let gridSize = MTLSize(width: imgWidth, height: imgHeight, depth: 1)
        let threadgroupSize = MTLSize(width: blockX, height: blockY, depth: 1)

        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        // Copy buffer to texture
        try copyBufferToTexture(
            commandBuffer: commandBuffer,
            buffer: outImg,
            texture: outputTexture,
            width: imgWidth,
            height: imgHeight,
            channels: Int(channels)
        )
    }

    // MARK: - Helper Functions

    private func countIntersects(numTilesHit: MTLBuffer, numPoints: Int) throws -> Int {
        // Copy buffer to CPU to sum up total intersections
        let sharedBuffer = device.makeBuffer(
            length: numPoints * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        )!

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw OpenSplatError.commandBufferCreationFailed
        }

        blitEncoder.copy(
            from: numTilesHit, sourceOffset: 0,
            to: sharedBuffer, destinationOffset: 0,
            size: numPoints * MemoryLayout<Int32>.stride
        )
        blitEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let ptr = sharedBuffer.contents().bindMemory(to: Int32.self, capacity: numPoints)
        var total: Int32 = 0
        for i in 0..<numPoints {
            total += ptr[i]
        }

        return Int(total)
    }

    private func copyBufferToTexture(
        commandBuffer: MTLCommandBuffer,
        buffer: MTLBuffer,
        texture: MTLTexture,
        width: Int,
        height: Int,
        channels: Int
    ) throws {
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            throw OpenSplatError.commandEncoderCreationFailed
        }

        let bytesPerRow = width * channels * MemoryLayout<Float>.stride
        let bytesPerImage = bytesPerRow * height

        blitEncoder.copy(
            from: buffer,
            sourceOffset: 0,
            sourceBytesPerRow: bytesPerRow,
            sourceBytesPerImage: bytesPerImage,
            sourceSize: MTLSize(width: width, height: height, depth: 1),
            to: texture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )

        blitEncoder.endEncoding()
    }
}
