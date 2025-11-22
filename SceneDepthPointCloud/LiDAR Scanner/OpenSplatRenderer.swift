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

        var allocatedCount: Int = 0   // Number of gaussians these buffers are sized for
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

        // Initialize GPU utilities
        self.gpuUtils = try GPUUtilities(device: device)

        print("✅ OpenSplatRenderer initialized with tile-based rasterization")
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
        guard count != intermediateBuffers.allocatedCount else { return }

        let tilesX = (imgWidth + Constants.blockX - 1) / Constants.blockX
        let tilesY = (imgHeight + Constants.blockY - 1) / Constants.blockY
        let numTiles = tilesX * tilesY

        // Allocate buffers (conservative estimate for intersections: count * 4 tiles per gaussian on average)
        let maxIntersections = count * 4

        intermediateBuffers.cov3d = device.makeBuffer(length: count * 6 * MemoryLayout<Float>.stride, options: .storageModePrivate)
        intermediateBuffers.xys = device.makeBuffer(length: count * 2 * MemoryLayout<Float>.stride, options: .storageModePrivate)
        intermediateBuffers.depths = device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModePrivate)
        intermediateBuffers.radii = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intermediateBuffers.conics = device.makeBuffer(length: count * 3 * MemoryLayout<Float>.stride, options: .storageModePrivate)
        intermediateBuffers.numTilesHit = device.makeBuffer(length: count * MemoryLayout<Int32>.stride, options: .storageModePrivate)

        intermediateBuffers.isectIds = device.makeBuffer(length: maxIntersections * MemoryLayout<Int64>.stride, options: .storageModePrivate)
        intermediateBuffers.gaussianIds = device.makeBuffer(length: maxIntersections * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intermediateBuffers.isectIdsSorted = device.makeBuffer(length: maxIntersections * MemoryLayout<Int64>.stride, options: .storageModePrivate)
        intermediateBuffers.gaussianIdsSorted = device.makeBuffer(length: maxIntersections * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        intermediateBuffers.tileBins = device.makeBuffer(length: numTiles * 2 * MemoryLayout<Int32>.stride, options: .storageModePrivate)

        // Also need indices buffer for tracking sort permutation
        intermediateBuffers.sortIndices = device.makeBuffer(length: maxIntersections * MemoryLayout<UInt32>.stride, options: .storageModePrivate)

        intermediateBuffers.allocatedCount = count

        print("📦 Allocated intermediate buffers for \(count) gaussians, \(numTiles) tiles")
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

        // Step 2: Bin gaussians to tiles and sort
        let numIntersections = binAndSortGaussians(
            commandBuffer: commandBuffer,
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
            commandBuffer: commandBuffer,
            gaussianData: gaussianData,
            imgWidth: imgWidth,
            imgHeight: imgHeight,
            outputTexture: outputTexture,
            background: background
        )

        commandBuffer.commit()
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

        // For now, use synchronous version to get the sum
        // In production, maintain sum on GPU or use indirect dispatch
        return gpuUtils.exclusivePrefixSumSync(input: input, output: input, count: count)
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

        // Output buffers (final_Ts, final_index, out_img)
        // For now, write directly to texture
        // TODO: Add proper output buffer management

        var bg = background
        encoder.setBytes(&bg, length: MemoryLayout<SIMD3<Float>>.stride, index: 9)

        var blockDim = SIMD2<UInt32>(UInt32(Constants.blockX), UInt32(Constants.blockY))
        encoder.setBytes(&blockDim, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 10)

        // Dispatch one threadgroup per tile
        let threadgroupSize = MTLSize(width: Constants.blockX, height: Constants.blockY, depth: 1)
        let threadgroups = MTLSize(width: tilesX, height: tilesY, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
