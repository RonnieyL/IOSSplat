import Foundation
import Metal
import MetalKit
import MetalPerformanceShaders
import simd

/**
 OpenSplat-based Gaussian Splatting Renderer

 This renderer uses OpenSplat's tile-based rasterization approach with GPU sorting.
 The rendering pipeline:
 1. Project gaussians to screen space (compute 2D covariance, bounding boxes)
 2. Bin gaussians to tiles (create intersection lists)
 3. Sort intersections by (tile_id, depth)
 4. Rasterize per-tile using sorted, culled lists
 */
public class OpenSplatRenderer {

    // MARK: - Constants

    private enum Constants {
        static let blockX: Int = 16  // Tile width (matches OpenSplat BLOCK_X)
        static let blockY: Int = 16  // Tile height (matches OpenSplat BLOCK_Y)
        static let clipThresh: Float = 0.01  // Near plane clipping threshold
    }

    // MARK: - Metal Resources

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // Compute pipeline states for OpenSplat kernels
    private let projectGaussiansForwardPipeline: MTLComputePipelineState
    private let mapGaussianToIntersectsPipeline: MTLComputePipelineState
    private let getTileBinEdgesPipeline: MTLComputePipelineState
    private let ndRasterizeForwardPipeline: MTLComputePipelineState
    private let computeShForwardPipeline: MTLComputePipelineState
    private let copyImageToTexturePipeline: MTLComputePipelineState

    // GPU utilities (prefix sum, sort, gather)
    private let gpuUtils: GPUUtilities

    // MARK: - Data Structures

    /// Gaussian data in OpenSplat format
    public struct GaussianData {
        var means3d: MTLBuffer        // float3 positions [N]
        var scales: MTLBuffer         // float3 scale per gaussian [N]
        var quats: MTLBuffer          // float4 rotation quaternion [N]
        var colors: MTLBuffer         // float3 RGB colors [N] (or SH coefficients)
        var opacities: MTLBuffer      // float opacity [N]
        var count: Int                // Number of gaussians
    }

    /// Intermediate buffers for rendering pipeline
    private struct IntermediateBuffers {
        var cov3d: MTLBuffer?         // float[6] 3D covariance (upper triangle) [N*6]
        var xys: MTLBuffer?           // float2 projected screen positions [N*2]
        var depths: MTLBuffer?        // float depth values [N]
        var radii: MTLBuffer?         // int radius in pixels [N]
        var conics: MTLBuffer?        // float3 2D covariance inverse [N*3]
        var numTilesHit: MTLBuffer?   // int32 tile count per gaussian [N]

        // Tile binning buffers
        var isectIds: MTLBuffer?      // int64 (tile_id << 32 | depth_bits) [totalIntersections]
        var gaussianIds: MTLBuffer?   // int32 gaussian index [totalIntersections]
        var isectIdsSorted: MTLBuffer?     // Sorted isectIds
        var gaussianIdsSorted: MTLBuffer?  // Sorted gaussianIds
        var tileBins: MTLBuffer?      // int2 (start, end) per tile [numTiles*2]
        var sortIndices: MTLBuffer?   // uint32 indices for sort permutation [totalIntersections]

        // Rasterization output buffers
        var finalTs: MTLBuffer?       // float final transmittance per pixel [numPixels]
        var finalIndex: MTLBuffer?    // int32 last gaussian index per pixel [numPixels]
        var outImg: MTLBuffer?        // float output image [numPixels * channels]
        
        // Intermediate texture for compute shader output (can't write directly to framebuffer-only drawable)
        var intermediateTexture: MTLTexture?

        var allocatedCount: Int = 0   // Number of gaussians these buffers are sized for
        var allocatedPixels: Int = 0  // Number of pixels these output buffers are sized for
    }

    private var intermediateBuffers = IntermediateBuffers()

    // MARK: - Initialization

    public init(device: MTLDevice) throws {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw NSError(domain: "OpenSplatRenderer", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = queue

        // Load the Metal library with OpenSplat kernels
        guard let library = try? device.makeDefaultLibrary() else {
            throw NSError(domain: "OpenSplatRenderer", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load Metal library"])
        }
        self.library = library

        // Create compute pipeline states
        self.projectGaussiansForwardPipeline = try Self.makePipeline(
            device: device, library: library, functionName: "project_gaussians_forward_kernel"
        )
        self.mapGaussianToIntersectsPipeline = try Self.makePipeline(
            device: device, library: library, functionName: "map_gaussian_to_intersects_kernel"
        )
        self.getTileBinEdgesPipeline = try Self.makePipeline(
            device: device, library: library, functionName: "get_tile_bin_edges_kernel"
        )
        self.ndRasterizeForwardPipeline = try Self.makePipeline(
            device: device, library: library, functionName: "nd_rasterize_forward_kernel"
        )
        self.computeShForwardPipeline = try Self.makePipeline(
            device: device, library: library, functionName: "compute_sh_forward_kernel"
        )
        self.copyImageToTexturePipeline = try Self.makePipeline(
            device: device, library: library, functionName: "copy_image_to_texture"
        )

        // Initialize GPU utilities
        self.gpuUtils = try GPUUtilities(device: device)

        print("✅ OpenSplatRenderer initialized with tile-based rasterization")
        print("   🔧 Radius clamping: MAX_RADIUS=32 pixels")
        print("   📦 Buffer allocation: 32 tiles per gaussian")
        print("   ⚙️  Version: 2024-11-22 (overflow fixes applied)")
    }

    private static func makePipeline(device: MTLDevice, library: MTLLibrary, functionName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(domain: "OpenSplatRenderer", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to find function: \(functionName)"])
        }
        return try device.makeComputePipelineState(function: function)
    }

    // MARK: - Buffer Management

    /// Allocate or reallocate intermediate buffers for a given gaussian count
    private func ensureIntermediateBuffers(count: Int, imgWidth: Int, imgHeight: Int) {
        let numPixels = imgWidth * imgHeight
        let needsRealloc = count != intermediateBuffers.allocatedCount || numPixels != intermediateBuffers.allocatedPixels
        guard needsRealloc else { return }

        let tilesX = (imgWidth + Constants.blockX - 1) / Constants.blockX
        let tilesY = (imgHeight + Constants.blockY - 1) / Constants.blockY
        let numTiles = tilesX * tilesY

        // Allocate buffers with much larger capacity for intersections
        // With MAX_RADIUS=32, gaussians hit ~40 tiles average (depends on position relative to tile grid)
        // Use 64 tiles per gaussian for safety margin
        let maxIntersections = count * 64
        
        print("📦 Allocating intersection buffers: \(maxIntersections) capacity (\(count) gaussians × 64 tiles)")
        print("   ⚠️ If you see this with old allocation numbers, REBUILD THE APP!")

        // Use shared mode for debugging (can read from CPU)
        intermediateBuffers.cov3d = device.makeBuffer(length: count * 6 * MemoryLayout<Float>.stride, options: .storageModeShared)
        intermediateBuffers.xys = device.makeBuffer(length: count * 2 * MemoryLayout<Float>.stride, options: .storageModeShared)
        intermediateBuffers.depths = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)
        intermediateBuffers.radii = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)
        intermediateBuffers.conics = device.makeBuffer(length: count * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)
        intermediateBuffers.numTilesHit = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModeShared)  // Changed to Shared for debugging

        intermediateBuffers.isectIds = device.makeBuffer(length: maxIntersections * MemoryLayout<Int64>.stride, options: .storageModePrivate)
        intermediateBuffers.gaussianIds = device.makeBuffer(length: maxIntersections * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intermediateBuffers.isectIdsSorted = device.makeBuffer(length: maxIntersections * MemoryLayout<Int64>.stride, options: .storageModePrivate)
        intermediateBuffers.gaussianIdsSorted = device.makeBuffer(length: maxIntersections * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intermediateBuffers.tileBins = device.makeBuffer(length: numTiles * 2 * MemoryLayout<Int32>.stride, options: .storageModePrivate)

        // Also need indices buffer for tracking sort permutation
        intermediateBuffers.sortIndices = device.makeBuffer(length: maxIntersections * MemoryLayout<UInt32>.stride, options: .storageModePrivate)

        // Allocate output buffers for rasterization
        intermediateBuffers.finalTs = device.makeBuffer(length: numPixels * MemoryLayout<Float>.stride, options: .storageModePrivate)
        intermediateBuffers.finalIndex = device.makeBuffer(length: numPixels * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intermediateBuffers.outImg = device.makeBuffer(length: numPixels * 3 * MemoryLayout<Float>.stride, options: .storageModePrivate)

        // Allocate intermediate texture (compute-writable, not framebuffer-only)
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: imgWidth,
            height: imgHeight,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        textureDescriptor.storageMode = .private
        intermediateBuffers.intermediateTexture = device.makeTexture(descriptor: textureDescriptor)

        intermediateBuffers.allocatedCount = count
        intermediateBuffers.allocatedPixels = numPixels

        print("📦 Allocated intermediate buffers for \(count) gaussians, \(numTiles) tiles, \(numPixels) pixels")
    }

    // MARK: - Rendering Pipeline

    /**
     Main rendering entry point

     - Parameters:
       - gaussianData: Gaussian parameters (positions, scales, rotations, colors, opacities)
       - viewMatrix: 4x4 view matrix (world to camera)
       - projMatrix: 4x4 projection matrix (camera to NDC)
       - intrinsics: Camera intrinsics (fx, fy, cx, cy)
       - imgWidth: Output image width
       - imgHeight: Output image height
       - outputTexture: Render target texture
       - background: Background color (RGB)
     */
    public func render(
        gaussianData: GaussianData,
        viewMatrix: simd_float4x4,
        projMatrix: simd_float4x4,
        intrinsics: SIMD4<Float>,  // (fx, fy, cx, cy)
        imgWidth: Int,
        imgHeight: Int,
        outputTexture: MTLTexture,
        background: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    ) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        commandBuffer.label = "OpenSplat Render"

        // Ensure intermediate buffers are allocated
        ensureIntermediateBuffers(count: gaussianData.count, imgWidth: imgWidth, imgHeight: imgHeight)

        // Step 1: Project gaussians to screen space
        projectGaussians(
            commandBuffer: commandBuffer,
            gaussianData: gaussianData,
            viewMatrix: viewMatrix,
            projMatrix: projMatrix,
            intrinsics: intrinsics,
            imgWidth: imgWidth,
            imgHeight: imgHeight
        )

        // DEBUG: Sync and check projection results
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        debugProjectionResults(gaussianData: gaussianData)

        // Create new command buffer for rest of pipeline
        guard let commandBuffer2 = commandQueue.makeCommandBuffer() else { return }
        commandBuffer2.label = "OpenSplat Render (Post Debug)"

        // Step 2: Bin gaussians to tiles and sort
        let numIntersections = binAndSortGaussians(
            commandBuffer: commandBuffer2,
            gaussianData: gaussianData,
            imgWidth: imgWidth,
            imgHeight: imgHeight
        )

        guard numIntersections > 0 else {
            print("⚠️ No gaussian intersections, skipping rasterization")
            return
        }

        // Step 3: Rasterize per-tile
        rasterize(
            commandBuffer: commandBuffer2,
            gaussianData: gaussianData,
            imgWidth: imgWidth,
            imgHeight: imgHeight,
            outputTexture: outputTexture,
            background: background
        )

        commandBuffer2.commit()
    }

    // MARK: - Step 1: Project Gaussians

    private func projectGaussians(
        commandBuffer: MTLCommandBuffer,
        gaussianData: GaussianData,
        viewMatrix: simd_float4x4,
        projMatrix: simd_float4x4,
        intrinsics: SIMD4<Float>,
        imgWidth: Int,
        imgHeight: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Project Gaussians"
        encoder.setComputePipelineState(projectGaussiansForwardPipeline)

        let tilesX = (imgWidth + Constants.blockX - 1) / Constants.blockX
        let tilesY = (imgHeight + Constants.blockY - 1) / Constants.blockY

        // Set buffers
        var numPoints = Int32(gaussianData.count)
        encoder.setBytes(&numPoints, length: MemoryLayout<Int32>.stride, index: 0)
        encoder.setBuffer(gaussianData.means3d, offset: 0, index: 1)
        encoder.setBuffer(gaussianData.scales, offset: 0, index: 2)

        var globScale: Float = 1.0
        encoder.setBytes(&globScale, length: MemoryLayout<Float>.stride, index: 3)
        encoder.setBuffer(gaussianData.quats, offset: 0, index: 4)

        var viewMat = viewMatrix
        var projMat = projMatrix
        var intrinsicsVar = intrinsics
        encoder.setBytes(&viewMat, length: MemoryLayout<simd_float4x4>.stride, index: 5)
        encoder.setBytes(&projMat, length: MemoryLayout<simd_float4x4>.stride, index: 6)
        encoder.setBytes(&intrinsicsVar, length: MemoryLayout<SIMD4<Float>>.stride, index: 7)

        var imgSize = SIMD2<UInt32>(UInt32(imgWidth), UInt32(imgHeight))
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 8)

        var tileBounds = SIMD3<UInt32>(UInt32(tilesX), UInt32(tilesY), 1)
        encoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 9)

        var clipThresh = Constants.clipThresh
        encoder.setBytes(&clipThresh, length: MemoryLayout<Float>.stride, index: 10)

        // Output buffers
        encoder.setBuffer(intermediateBuffers.cov3d, offset: 0, index: 11)
        encoder.setBuffer(intermediateBuffers.xys, offset: 0, index: 12)
        encoder.setBuffer(intermediateBuffers.depths, offset: 0, index: 13)
        encoder.setBuffer(intermediateBuffers.radii, offset: 0, index: 14)
        encoder.setBuffer(intermediateBuffers.conics, offset: 0, index: 15)
        encoder.setBuffer(intermediateBuffers.numTilesHit, offset: 0, index: 16)

        // Dispatch
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (gaussianData.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        // DEBUG: Log projection parameters
        print("🔍 DEBUG Project: \(gaussianData.count) gaussians, img=\(imgWidth)x\(imgHeight), tiles=\(tilesX)x\(tilesY)")
        print("   Intrinsics: fx=\(intrinsics.x), fy=\(intrinsics.y), cx=\(intrinsics.z), cy=\(intrinsics.w)")
        print("   Clip thresh: \(clipThresh)")
    }

    // MARK: - Debug Helper

    private func debugProjectionResults(gaussianData: GaussianData) {
        // DEBUG: Read unpacked gaussian data (first few gaussians)
        // IMPORTANT: Buffers are packed float arrays, NOT SIMD types!
        // means3d = [x0, y0, z0, x1, y1, z1, ...] (12 bytes per gaussian)
        // SIMD3<Float> is 16-byte aligned, which would cause wrong offsets!
        let means3dPtr = gaussianData.means3d.contents().bindMemory(to: Float.self, capacity: gaussianData.count * 3)
        let scalesPtr = gaussianData.scales.contents().bindMemory(to: Float.self, capacity: gaussianData.count * 3)
        let quatsPtr = gaussianData.quats.contents().bindMemory(to: Float.self, capacity: gaussianData.count * 4)

        print("🔍 DEBUG: Unpacked buffer data (first 3 gaussians):")
        for i in 0..<min(3, gaussianData.count) {
            let pos = SIMD3<Float>(means3dPtr[i * 3 + 0], means3dPtr[i * 3 + 1], means3dPtr[i * 3 + 2])
            let scale = SIMD3<Float>(scalesPtr[i * 3 + 0], scalesPtr[i * 3 + 1], scalesPtr[i * 3 + 2])
            let quat = SIMD4<Float>(quatsPtr[i * 4 + 0], quatsPtr[i * 4 + 1], quatsPtr[i * 4 + 2], quatsPtr[i * 4 + 3])
            print("  [\(i)] pos=(\(String(format: "%.3f", pos.x)), \(String(format: "%.3f", pos.y)), \(String(format: "%.3f", pos.z)))")
            print("       scale=(\(String(format: "%.4f", scale.x)), \(String(format: "%.4f", scale.y)), \(String(format: "%.4f", scale.z)))")
            print("       quat=(\(String(format: "%.3f", quat.x)), \(String(format: "%.3f", quat.y)), \(String(format: "%.3f", quat.z)), \(String(format: "%.3f", quat.w)))")
        }

        // DEBUG: Read projection outputs (xys, depths, radii)
        let xysPtr = intermediateBuffers.xys!.contents().bindMemory(to: SIMD2<Float>.self, capacity: gaussianData.count)
        let depthsPtr = intermediateBuffers.depths!.contents().bindMemory(to: Float.self, capacity: gaussianData.count)
        let radiiPtr = intermediateBuffers.radii!.contents().bindMemory(to: Int32.self, capacity: gaussianData.count)

        print("🔍 DEBUG: Projection outputs (first 10 gaussians):")
        for i in 0..<min(10, gaussianData.count) {
            let xy = xysPtr[i]
            let depth = depthsPtr[i]
            let radius = radiiPtr[i]

            let status: String
            if radius == -1 {
                status = "CLIPPED (near plane, z=\(String(format: "%.3f", depth)) < 0.01)"
            } else if radius == -2 {
                status = "FAILED (zero determinant covariance)"
            } else if radius == -3 {
                status = "NO TILES (off-screen or zero radius)"
            } else if radius > 0 {
                status = "SUCCESS"
            } else {
                status = "UNKNOWN"
            }

            print("  [\(i)] \(status): xy=(\(String(format: "%.1f", xy.x)), \(String(format: "%.1f", xy.y))), depth=\(String(format: "%.3f", depth)), radius=\(radius)")
        }
    }

    // MARK: - Step 2: Bin and Sort Gaussians

    private func binAndSortGaussians(
        commandBuffer: MTLCommandBuffer,
        gaussianData: GaussianData,
        imgWidth: Int,
        imgHeight: Int
    ) -> Int {
        // Step 2a: Prefix sum on numTilesHit to get cumulative offsets
        let numIntersections = computePrefixSum(
            commandBuffer: commandBuffer,
            input: intermediateBuffers.numTilesHit!,
            count: gaussianData.count
        )

        guard numIntersections > 0 else { return 0 }

        // Step 2b: Map gaussians to tile intersections
        mapGaussianToIntersects(
            commandBuffer: commandBuffer,
            gaussianData: gaussianData,
            imgWidth: imgWidth,
            imgHeight: imgHeight,
            numIntersections: numIntersections
        )

        // Step 2c: Sort intersections by (tile_id, depth) using MPS RadixSort
        sortIntersections(
            commandBuffer: commandBuffer,
            numIntersections: numIntersections
        )

        // Step 2d: Find tile bin edges (start/end indices for each tile)
        getTileBinEdges(
            commandBuffer: commandBuffer,
            numIntersections: numIntersections
        )

        return numIntersections
    }

    private func computePrefixSum(commandBuffer: MTLCommandBuffer, input: MTLBuffer, count: Int) -> Int {
        // Use GPUUtilities to compute exclusive prefix sum
        // This also returns the total sum (total number of intersections)

        // DEBUG: Read numTilesHit buffer BEFORE prefix sum to see rejection reasons
        // Need to wait for projection kernel to complete
        guard let syncBuffer = commandQueue.makeCommandBuffer() else { return 0 }
        syncBuffer.commit()
        syncBuffer.waitUntilCompleted()

        let ptr = input.contents().bindMemory(to: Int32.self, capacity: count)
        var zeroCount = 0
        var nonZeroCount = 0
        var samples: [Int32] = []
        for i in 0..<min(100, count) {
            let tiles = ptr[i]
            if tiles == 0 {
                zeroCount += 1
            } else {
                nonZeroCount += 1
                if samples.count < 10 {
                    samples.append(tiles)
                }
            }
        }
        print("🔍 DEBUG: numTilesHit BEFORE prefix sum (first 100):")
        print("   Zero: \(zeroCount), Non-zero: \(nonZeroCount)")
        if !samples.isEmpty {
            print("   Non-zero samples: \(samples)")
        } else {
            print("   ⚠️ ALL ZEROS - all gaussians rejected by projection kernel!")
        }

        // For now, use synchronous version to get the sum
        // In production, maintain sum on GPU or use indirect dispatch
        let total = gpuUtils.exclusivePrefixSumSync(input: input, output: input, count: count)

        // DEBUG: Log intersection count and check for buffer overflow
        let maxCapacity = count * 64  // Must match allocation in ensureIntermediateBuffers
        let avgPerGaussian = Double(total) / Double(count)
        let utilizationPercent = (Double(total) / Double(maxCapacity)) * 100.0
        print("🔍 DEBUG: Prefix sum - \(total) intersections from \(count) gaussians")
        print("   Average: \(String(format: "%.1f", avgPerGaussian)) intersections/gaussian")
        print("   Buffer utilization: \(String(format: "%.1f", utilizationPercent))% of \(maxCapacity) capacity")
        
        if total > maxCapacity {
            print("⚠️ BUFFER OVERFLOW: \(total) intersections exceeds capacity \(maxCapacity)!")
            print("   This will cause artifacts. Increase maxIntersections or reduce gaussian scale.")
            // Clamp to prevent crash, but rendering will be incorrect
            return maxCapacity
        }
        
        if utilizationPercent > 80.0 {
            print("⚠️ Buffer near capacity (\(String(format: "%.1f", utilizationPercent))%). Consider increasing allocation.")
        }
        
        return total
    }

    private func mapGaussianToIntersects(
        commandBuffer: MTLCommandBuffer,
        gaussianData: GaussianData,
        imgWidth: Int,
        imgHeight: Int,
        numIntersections: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Map Gaussian to Intersects"
        encoder.setComputePipelineState(mapGaussianToIntersectsPipeline)

        let tilesX = (imgWidth + Constants.blockX - 1) / Constants.blockX
        let tilesY = (imgHeight + Constants.blockY - 1) / Constants.blockY

        var numPoints = Int32(gaussianData.count)
        encoder.setBytes(&numPoints, length: MemoryLayout<Int32>.stride, index: 0)
        encoder.setBuffer(intermediateBuffers.xys, offset: 0, index: 1)
        encoder.setBuffer(intermediateBuffers.depths, offset: 0, index: 2)
        encoder.setBuffer(intermediateBuffers.radii, offset: 0, index: 3)
        encoder.setBuffer(intermediateBuffers.numTilesHit, offset: 0, index: 4)  // This should be cumulative sum

        var tileBounds = SIMD3<UInt32>(UInt32(tilesX), UInt32(tilesY), 1)
        encoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 5)

        encoder.setBuffer(intermediateBuffers.isectIds, offset: 0, index: 6)
        encoder.setBuffer(intermediateBuffers.gaussianIds, offset: 0, index: 7)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (gaussianData.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    private func sortIntersections(commandBuffer: MTLCommandBuffer, numIntersections: Int) {
        // Sort isectIds using GPU radix sort with index tracking
        gpuUtils.radixSortInt64WithIndices(
            commandBuffer: commandBuffer,
            keys: intermediateBuffers.isectIds!,
            keysSorted: intermediateBuffers.isectIdsSorted!,
            indices: intermediateBuffers.sortIndices!,
            count: numIntersections
        )

        // Reorder gaussianIds based on sorted indices
        gpuUtils.gatherInt32(
            commandBuffer: commandBuffer,
            indices: intermediateBuffers.sortIndices!,
            source: intermediateBuffers.gaussianIds!,
            destination: intermediateBuffers.gaussianIdsSorted!,
            count: numIntersections
        )
    }

    private func getTileBinEdges(commandBuffer: MTLCommandBuffer, numIntersections: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Get Tile Bin Edges"
        encoder.setComputePipelineState(getTileBinEdgesPipeline)

        var numIntersects = Int32(numIntersections)
        encoder.setBytes(&numIntersects, length: MemoryLayout<Int32>.stride, index: 0)
        encoder.setBuffer(intermediateBuffers.isectIdsSorted, offset: 0, index: 1)
        encoder.setBuffer(intermediateBuffers.tileBins, offset: 0, index: 2)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numIntersections + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    // MARK: - Step 3: Rasterize

    private func rasterize(
        commandBuffer: MTLCommandBuffer,
        gaussianData: GaussianData,
        imgWidth: Int,
        imgHeight: Int,
        outputTexture: MTLTexture,
        background: SIMD3<Float>
    ) {
        // Clear output buffer before rasterization (important!)
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
        blitEncoder.label = "Clear Output Buffer"
        let numPixels = imgWidth * imgHeight
        blitEncoder.fill(buffer: intermediateBuffers.outImg!, range: 0..<(numPixels * 3 * MemoryLayout<Float>.stride), value: 0)
        blitEncoder.endEncoding()
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Rasterize Gaussians"
        encoder.setComputePipelineState(ndRasterizeForwardPipeline)

        let tilesX = (imgWidth + Constants.blockX - 1) / Constants.blockX
        let tilesY = (imgHeight + Constants.blockY - 1) / Constants.blockY

        var tileBounds = SIMD3<UInt32>(UInt32(tilesX), UInt32(tilesY), 1)
        encoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 0)

        var imgSize = SIMD3<UInt32>(UInt32(imgWidth), UInt32(imgHeight), 1)
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD3<UInt32>>.stride, index: 1)

        var channels: UInt32 = 3
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.stride, index: 2)

        encoder.setBuffer(intermediateBuffers.gaussianIdsSorted, offset: 0, index: 3)
        encoder.setBuffer(intermediateBuffers.tileBins, offset: 0, index: 4)
        encoder.setBuffer(intermediateBuffers.xys, offset: 0, index: 5)
        encoder.setBuffer(intermediateBuffers.conics, offset: 0, index: 6)
        encoder.setBuffer(gaussianData.colors, offset: 0, index: 7)
        encoder.setBuffer(gaussianData.opacities, offset: 0, index: 8)

        // Output buffers (final_Ts, final_index, out_img) - REQUIRED by shader!
        encoder.setBuffer(intermediateBuffers.finalTs, offset: 0, index: 9)
        encoder.setBuffer(intermediateBuffers.finalIndex, offset: 0, index: 10)
        encoder.setBuffer(intermediateBuffers.outImg, offset: 0, index: 11)

        var bg = background
        encoder.setBytes(&bg, length: MemoryLayout<SIMD3<Float>>.stride, index: 12)

        var blockDim = SIMD2<UInt32>(UInt32(Constants.blockX), UInt32(Constants.blockY))
        encoder.setBytes(&blockDim, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 13)

        // Dispatch one threadgroup per tile
        let threadgroupSize = MTLSize(width: Constants.blockX, height: Constants.blockY, depth: 1)
        let threadgroups = MTLSize(width: tilesX, height: tilesY, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        // Copy rendered image buffer to intermediate texture (compute-writable)
        guard let intermediateTexture = intermediateBuffers.intermediateTexture else {
            print("⚠️ No intermediate texture available")
            return
        }
        
        guard let copyEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        copyEncoder.label = "Copy Image to Texture"
        copyEncoder.setComputePipelineState(copyImageToTexturePipeline)
        
        copyEncoder.setBuffer(intermediateBuffers.outImg, offset: 0, index: 0)
        copyEncoder.setTexture(intermediateTexture, index: 0)
        
        let copyThreadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let copyThreadgroups = MTLSize(
            width: (imgWidth + copyThreadgroupSize.width - 1) / copyThreadgroupSize.width,
            height: (imgHeight + copyThreadgroupSize.height - 1) / copyThreadgroupSize.height,
            depth: 1
        )
        copyEncoder.dispatchThreadgroups(copyThreadgroups, threadsPerThreadgroup: copyThreadgroupSize)
        copyEncoder.endEncoding()
        
        // Blit intermediate texture to drawable (framebuffer-only) texture
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
        blitEncoder.label = "Blit to Drawable"
        blitEncoder.copy(
            from: intermediateTexture,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: imgWidth, height: imgHeight, depth: 1),
            to: outputTexture,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        blitEncoder.endEncoding()
        
        print("✅ Rasterization complete, image blitted to drawable")
    }
}
