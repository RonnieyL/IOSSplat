import Metal
import MetalKit

class GaussianSplatRenderer {
    // MARK: - Pipeline States
    private var projectPipeline: MTLComputePipelineState!
    private var mapIntersectsPipeline: MTLComputePipelineState!
    private var sortPipeline: MTLComputePipelineState! // Bitonic or Radix
    private var tileBinPipeline: MTLComputePipelineState!
    private var rasterizePipeline: MTLComputePipelineState!
    private var displayPipeline: MTLComputePipelineState!
    private var clearBufferPipeline: MTLComputePipelineState!
    
    // Helper Classes
    private var prefixSum: PrefixSum!
    private var bitonicSort: BitonicSort!
    
    private var pointCount: Int = 0
    
    // MARK: - Buffers
    // Gaussian Data (Persistent)
    var meansBuffer: MTLBuffer!      // 3D Positions
    var scalesBuffer: MTLBuffer!     // 3D Scales
    var quatsBuffer: MTLBuffer!      // Rotations
    var opacitiesBuffer: MTLBuffer!  // Alpha
    var colorsBuffer: MTLBuffer!     // SH Coeffs or RGB
    
    // Intermediate Data (Per Frame)
    private var cov3dBuffer: MTLBuffer!
    private var xysBuffer: MTLBuffer!
    private var depthsBuffer: MTLBuffer!
    private var radiiBuffer: MTLBuffer!
    private var conicsBuffer: MTLBuffer!
    private var numTilesHitBuffer: MTLBuffer! // Used for Counts -> then Offsets
    private var imgBuffer: MTLBuffer! // Intermediate image buffer
    
    // Sorting Data
    private var isectIdsBuffer: MTLBuffer!    // (TileID << 32) | Depth
    private var gaussianIdsBuffer: MTLBuffer! // Splat Index
    private var tileBinsBuffer: MTLBuffer!    // [Start, End] per tile
    
    // MARK: - Initialization
    init(device: MTLDevice) {
        // 1. Load Library
        // 2. Create Pipeline States for all kernels
        // 3. Allocate initial buffers (can resize dynamically)
        let library = device.makeDefaultLibrary()!
                
                // Create compute pipeline states
                projectPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "project_gaussians_forward_kernel")!)
                mapIntersectsPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "map_gaussian_to_intersects_kernel")!)
                sortPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "bitonic_sort_kernel")!)
                tileBinPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "get_tile_bin_edges_kernel")!)
                rasterizePipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "nd_rasterize_forward_kernel")!)
                displayPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "display_kernel")!)
                clearBufferPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "clear_buffer_kernel")!)
                
                // Initialize Helper Classes
                prefixSum = PrefixSum(device: device)
                bitonicSort = BitonicSort(device: device)
                
                // Allocate initial buffers with reasonable default sizes
                let initialMaxPoints = 1000000
                let initialMaxIntersects = 100000
                
                // Gaussian data buffers
                meansBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
                scalesBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
                quatsBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)
                opacitiesBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<Float>.stride, options: .storageModeShared)
                colorsBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
                
                // Intermediate buffers
                cov3dBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD3<Float>>.stride * 2, options: .storageModePrivate)
                xysBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD2<Float>>.stride, options: .storageModePrivate)
                depthsBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<Float>.stride, options: .storageModePrivate)
                radiiBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<Int32>.stride, options: .storageModePrivate)
                conicsBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
                numTilesHitBuffer = device.makeBuffer(length: initialMaxPoints * MemoryLayout<Int32>.stride, options: .storageModeShared)
                
                // Sorting buffers
                isectIdsBuffer = device.makeBuffer(length: initialMaxIntersects * MemoryLayout<UInt64>.stride, options: .storageModePrivate)
                gaussianIdsBuffer = device.makeBuffer(length: initialMaxIntersects * MemoryLayout<Int32>.stride, options: .storageModePrivate)
                
                // Tile bins buffer (assuming 16x16 tiles for typical screen)
                let maxTiles = 256 * 256
                tileBinsBuffer = device.makeBuffer(length: maxTiles * MemoryLayout<SIMD2<Int32>>.stride, options: .storageModePrivate)
                
                // Image buffer (assuming max 4K resolution for now, resize in draw if needed)
                let maxPixels = 3840 * 2160
                imgBuffer = device.makeBuffer(length: maxPixels * 3 * MemoryLayout<Float>.stride, options: .storageModePrivate)
    }
    
    // MARK: - Data Loading
    func addPoints(positions: [SIMD3<Float>], colors: [SIMD3<Float>], covariances: [SIMD3<Float>]) {
        // 1. Append new data to meansBuffer, colorsBuffer
        // 2. Initialize scales (e.g., 0.01), quats (Identity), opacities (1.0)
        // 3. Resize intermediate buffers if maxPoints exceeded
        guard positions.count == colors.count && positions.count == covariances.count else {
            print("Error: positions, colors, and covariances arrays must have the same count")
            return
        }
        
        let newPointCount = positions.count
        guard newPointCount > 0 else { return }
        
        // Get current point count from buffer capacity
        let currentCapacity = meansBuffer.length / MemoryLayout<SIMD3<Float>>.stride
        let device = meansBuffer.device
        
        // Resize buffers if needed
        if newPointCount > currentCapacity {
            let newCapacity = max(newPointCount, currentCapacity * 2)
            
            // Reallocate Gaussian data buffers
            meansBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            scalesBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            quatsBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)
            opacitiesBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<Float>.stride, options: .storageModeShared)
            colorsBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            
            // Reallocate intermediate buffers
            cov3dBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride * 2, options: .storageModePrivate)
            xysBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD2<Float>>.stride, options: .storageModePrivate)
            depthsBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<Float>.stride, options: .storageModePrivate)
            radiiBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<Int32>.stride, options: .storageModePrivate)
            conicsBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModePrivate)
            numTilesHitBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<Int32>.stride, options: .storageModeShared)
            
            // Reallocate sorting buffers with larger intersection capacity
            // Ensure capacity is Power of Two for Bitonic Sort
            let rawIntersectCapacity = newCapacity * 10 // Assume avg 10 tiles per splat
            var potIntersectCapacity = 1
            while potIntersectCapacity < rawIntersectCapacity { potIntersectCapacity <<= 1 }
            
            isectIdsBuffer = device.makeBuffer(length: potIntersectCapacity * MemoryLayout<UInt64>.stride, options: .storageModePrivate)
            gaussianIdsBuffer = device.makeBuffer(length: potIntersectCapacity * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        }
        
        // Copy position data
        let meansPointer = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: newPointCount)
        for i in 0..<newPointCount {
            meansPointer[i] = positions[i]
        }
        
        // Copy color data
        let colorsPointer = colorsBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: newPointCount)
        for i in 0..<newPointCount {
            colorsPointer[i] = colors[i]
        }
        
        // Copy covariance data (assuming this represents scale components) might change this to taking an input of the probability and 
        let scalesPointer = scalesBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: newPointCount)
        for i in 0..<newPointCount {
            scalesPointer[i] = covariances[i]
        }
        
        // Initialize quaternions to identity rotation
        let quatsPointer = quatsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: newPointCount)
        for i in 0..<newPointCount {
            quatsPointer[i] = SIMD4<Float>(0, 0, 0, 1) // Identity quaternion (x, y, z, w)
        }
        
        // Initialize opacities to fully opaque
        let opacitiesPointer = opacitiesBuffer.contents().bindMemory(to: Float.self, capacity: newPointCount)
        for i in 0..<newPointCount {
            opacitiesPointer[i] = 1.0
        }
        
        self.pointCount = newPointCount
    }
    
    
    // MARK: - Draw Loop
    func draw(commandBuffer: MTLCommandBuffer, 
              viewMatrix: matrix_float4x4, 
              projectionMatrix: matrix_float4x4, 
              viewport: CGSize, 
              outputTexture: MTLTexture) {
        
        // 1. Projection Pass
        // ------------------
        // Calculates 2D attributes and how many tiles each splat hits.
        // Kernel: project_gaussians_forward_kernel
        // Writes to: num_tiles_hitBuffer (as Counts)
        dispatchProjection(commandBuffer: commandBuffer, 
                           viewMatrix: viewMatrix, 
                           projectionMatrix: projectionMatrix, 
                           viewport: viewport)
        
        // 2. Prefix Sum (Scan) Pass
        // -------------------------
        // Converts "Counts" to "Offsets" so we know where to write in the sort buffer.
        // Input: num_tiles_hitBuffer (Counts)
        // Output: num_tiles_hitBuffer (Offsets)
        // Note: Can be CPU roundtrip for MVP, or GPU Blelloch Scan for performance.
        performPrefixSum(commandBuffer: commandBuffer)
        
        // 3. Binning Pass
        // ---------------
        // Populates the sort keys based on the offsets calculated above.
        // Kernel: map_gaussian_to_intersects_kernel
        // Input: num_tiles_hitBuffer (Offsets)
        // Writes to: isectIdsBuffer, gaussianIdsBuffer
        
        // Initialize sort buffer to UINT64_MAX before binning so unused slots sort to end
        let capacity = isectIdsBuffer.length / MemoryLayout<UInt64>.stride
        bitonicSort.pad(commandBuffer: commandBuffer, keys: isectIdsBuffer, startIndex: 0, endIndex: capacity)
        
        dispatchBinning(commandBuffer: commandBuffer, viewport: viewport)
        
        // 4. Sorting Pass
        // ---------------
        // Sorts the splats by TileID (primary) and Depth (secondary).
        // Kernel: Bitonic Sort (multiple dispatches)
        // Sorts: isectIdsBuffer and gaussianIdsBuffer in place
        dispatchSort(commandBuffer: commandBuffer, ...)
        
        // 5. Tile Range Pass
        // ------------------
        // Identifies the start and end indices for each tile in the sorted list.
        // Kernel: get_tile_bin_edges_kernel
        // Writes to: tileBinsBuffer
        dispatchTileBinning(commandBuffer: commandBuffer, ...)
        
        // 6. Rasterization Pass
        // ---------------------
        // The heavy lifter. Draws the sorted splats tile-by-tile.
        // Kernel: nd_rasterize_forward_kernel
        // Input: tileBinsBuffer, sorted gaussianIdsBuffer
        // Output: Intermediate Image Buffer
        dispatchRasterization(commandBuffer: commandBuffer, viewport: viewport)
        
        // 7. Display Pass
        // ---------------
        // Copies the raw float buffer to the drawable texture.
        dispatchDisplay(commandBuffer: commandBuffer, texture: outputTexture)
    }
    
    // MARK: - Helper Dispatch Functions
    
    private func dispatchProjection(commandBuffer: MTLCommandBuffer, 
                                    viewMatrix: matrix_float4x4, 
                                    projectionMatrix: matrix_float4x4, 
                                    viewport: CGSize) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Projection Pass"
        encoder.setComputePipelineState(projectPipeline)
        
        var numPoints = Int32(pointCount)
        var globScale: Float = 1.0 // TODO: Expose this
        var clipThresh: Float = 0.2
        var imgSize = SIMD2<UInt32>(UInt32(viewport.width), UInt32(viewport.height))
        var tileBounds = SIMD3<UInt32>((UInt32(viewport.width) + 15) / 16, (UInt32(viewport.height) + 15) / 16, 1)
        
        // Set Buffers
        encoder.setBytes(&numPoints, length: MemoryLayout<Int32>.size, index: 0)
        encoder.setBuffer(meansBuffer, offset: 0, index: 1)
        encoder.setBuffer(scalesBuffer, offset: 0, index: 2)
        encoder.setBytes(&globScale, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBuffer(quatsBuffer, offset: 0, index: 4)
        
        var viewMat = viewMatrix
        var projMat = projectionMatrix
        encoder.setBytes(&viewMat, length: MemoryLayout<matrix_float4x4>.size, index: 5)
        encoder.setBytes(&projMat, length: MemoryLayout<matrix_float4x4>.size, index: 6)
        
        // Intrinsics (Approximation from projection matrix)
        let fx = projectionMatrix.columns.0.x * Float(viewport.width) / 2.0
        let fy = projectionMatrix.columns.1.y * Float(viewport.height) / 2.0
        let cx = Float(viewport.width) / 2.0
        let cy = Float(viewport.height) / 2.0
        var intrins = SIMD4<Float>(fx, fy, cx, cy)
        encoder.setBytes(&intrins, length: MemoryLayout<SIMD4<Float>>.size, index: 7)
        
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 8)
        encoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 9)
        encoder.setBytes(&clipThresh, length: MemoryLayout<Float>.size, index: 10)
        
        encoder.setBuffer(cov3dBuffer, offset: 0, index: 11)
        encoder.setBuffer(xysBuffer, offset: 0, index: 12)
        encoder.setBuffer(depthsBuffer, offset: 0, index: 13)
        encoder.setBuffer(radiiBuffer, offset: 0, index: 14)
        encoder.setBuffer(conicsBuffer, offset: 0, index: 15)
        encoder.setBuffer(numTilesHitBuffer, offset: 0, index: 16)
        
        let gridSize = MTLSize(width: pointCount, height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: min(projectPipeline.maxTotalThreadsPerThreadgroup, pointCount), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
    }
    
    private func performPrefixSum(commandBuffer: MTLCommandBuffer) {
        // GPU Parallel Scan using PrefixSum helper class
        prefixSum.compute(commandBuffer: commandBuffer, 
                          inputBuffer: numTilesHitBuffer, 
                          outputBuffer: numTilesHitBuffer, 
                          count: pointCount)
    }
    
    private func dispatchBinning(commandBuffer: MTLCommandBuffer, viewport: CGSize) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Binning Pass"
        encoder.setComputePipelineState(mapIntersectsPipeline)
        
        var numPoints = Int32(pointCount)
        var tileBounds = SIMD3<UInt32>((UInt32(viewport.width) + 15) / 16, (UInt32(viewport.height) + 15) / 16, 1)
        
        encoder.setBytes(&numPoints, length: MemoryLayout<Int32>.size, index: 0)
        encoder.setBuffer(xysBuffer, offset: 0, index: 1)
        encoder.setBuffer(depthsBuffer, offset: 0, index: 2)
        encoder.setBuffer(radiiBuffer, offset: 0, index: 3)
        encoder.setBuffer(numTilesHitBuffer, offset: 0, index: 4) // Now contains offsets
        encoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 5)
        encoder.setBuffer(isectIdsBuffer, offset: 0, index: 6)
        encoder.setBuffer(gaussianIdsBuffer, offset: 0, index: 7)
        
        let gridSize = MTLSize(width: pointCount, height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: min(mapIntersectsPipeline.maxTotalThreadsPerThreadgroup, pointCount), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
    }
    
    private func dispatchSort(commandBuffer: MTLCommandBuffer) {
        // Bitonic Sort Logic:
        // 1. Pad the unused portion of the buffer with UINT64_MAX
        // 2. Sort the entire power-of-two buffer
        
        let capacity = isectIdsBuffer.length / MemoryLayout<UInt64>.stride
        
        bitonicSort.sort(commandBuffer: commandBuffer, 
                         keys: isectIdsBuffer, 
                         values: gaussianIdsBuffer, 
                         count: capacity)
    }
    
    private func dispatchTileBinning(commandBuffer: MTLCommandBuffer) {
        // Clear tile bins first
        guard let clearEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        clearEncoder.label = "Clear Tile Bins"
        clearEncoder.setComputePipelineState(clearBufferPipeline)
        clearEncoder.setBuffer(tileBinsBuffer, offset: 0, index: 0)
        var zero: Int32 = 0 // Or -1 depending on logic, using 0 for now as default
        clearEncoder.setBytes(&zero, length: MemoryLayout<Int32>.size, index: 1)
        let clearGrid = MTLSize(width: tileBinsBuffer.length / MemoryLayout<Int32>.stride, height: 1, depth: 1)
        let clearThreads = MTLSize(width: min(clearBufferPipeline.maxTotalThreadsPerThreadgroup, clearGrid.width), height: 1, depth: 1)
        clearEncoder.dispatchThreads(clearGrid, threadsPerThreadgroup: clearThreads)
        clearEncoder.endEncoding()
        
        // Bin Edges
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Tile Bin Edges"
        encoder.setComputePipelineState(tileBinPipeline)
        
        // We use the full capacity because we padded with UINT64_MAX
        var numIntersects = Int32(isectIdsBuffer.length / MemoryLayout<UInt64>.stride)
        
        encoder.setBytes(&numIntersects, length: MemoryLayout<Int32>.size, index: 0)
        encoder.setBuffer(isectIdsBuffer, offset: 0, index: 1)
        encoder.setBuffer(tileBinsBuffer, offset: 0, index: 2)
        
        let gridSize = MTLSize(width: Int(numIntersects), height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: min(tileBinPipeline.maxTotalThreadsPerThreadgroup, Int(numIntersects)), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
    }
    
    private func dispatchDisplay(commandBuffer: MTLCommandBuffer, texture: MTLTexture) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Display Pass"
        encoder.setComputePipelineState(displayPipeline)
        
        encoder.setBuffer(imgBuffer, offset: 0, index: 0)
        encoder.setTexture(texture, index: 0)
        
        var imgSize = SIMD2<UInt32>(UInt32(texture.width), UInt32(texture.height))
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        
        let gridSize = MTLSize(width: texture.width, height: texture.height, depth: 1)
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1) // Standard 2D block
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
    }

    private func dispatchRasterization(commandBuffer: MTLCommandBuffer, viewport: CGSize) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Rasterization Pass"
        encoder.setComputePipelineState(rasterizePipeline)
        
        var tileBounds = SIMD3<UInt32>((UInt32(viewport.width) + 15) / 16, (UInt32(viewport.height) + 15) / 16, 1)
        var imgSize = SIMD3<UInt32>(UInt32(viewport.width), UInt32(viewport.height), 1)
        var channels: UInt32 = 3
        var background = SIMD3<Float>(0, 0, 0) // Black background
        
        encoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 0)
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD3<UInt32>>.size, index: 1)
        encoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 2)
        
        encoder.setBuffer(gaussianIdsBuffer, offset: 0, index: 3)
        encoder.setBuffer(tileBinsBuffer, offset: 0, index: 4)
        encoder.setBuffer(xysBuffer, offset: 0, index: 5)
        encoder.setBuffer(conicsBuffer, offset: 0, index: 6)
        encoder.setBuffer(colorsBuffer, offset: 0, index: 7)
        encoder.setBuffer(opacitiesBuffer, offset: 0, index: 8)
        
        // These are optional/debug outputs in the kernel signature, can be nil or dummy buffers if not used
        // But Metal requires binding if they are in the argument table.
        // Let's create small dummy buffers for now or reuse existing scratch if safe.
        // Actually, looking at kernel: device float* final_Ts, device int* final_index
        // We should allocate them.
        let pixelCount = Int(viewport.width * viewport.height)
        let finalTsBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.stride, options: .storageModePrivate)
        let finalIndexBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        
        encoder.setBuffer(finalTsBuffer, offset: 0, index: 9)
        encoder.setBuffer(finalIndexBuffer, offset: 0, index: 10)
        encoder.setBuffer(imgBuffer, offset: 0, index: 11)
        encoder.setBytes(&background, length: MemoryLayout<SIMD3<Float>>.size, index: 12)
        
        // Threadgroup config
        let blockDim = SIMD2<UInt32>(16, 16)
        encoder.setBytes(&blockDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 13)
        
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(width: Int(tileBounds.x), height: Int(tileBounds.y), depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
    }
}