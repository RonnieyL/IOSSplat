import Metal
import MetalKit

class GaussianSplatRenderer {
    // MARK: - Device
    private let device: MTLDevice
    
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
    
    // Rasterization scratch buffers
    private var finalTsBuffer: MTLBuffer!
    private var finalIndexBuffer: MTLBuffer!
    
    // Intermediate texture for compute output (drawable textures are frameBufferOnly)
    private var intermediateTexture: MTLTexture?
    private var copyPipeline: MTLRenderPipelineState?
    
    // Sorting Data
    private var isectIdsBuffer: MTLBuffer!    // (TileID << 32) | Depth
    private var gaussianIdsBuffer: MTLBuffer! // Splat Index
    private var tileBinsBuffer: MTLBuffer!    // [Start, End] per tile
    
    // MARK: - Initialization
    init(device: MTLDevice) {
        self.device = device
        
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
                
                // Create render pipeline for copying intermediate texture to drawable
                let copyPipelineDescriptor = MTLRenderPipelineDescriptor()
                copyPipelineDescriptor.vertexFunction = library.makeFunction(name: "copyVertex")
                copyPipelineDescriptor.fragmentFunction = library.makeFunction(name: "copyFragment")
                copyPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
                copyPipeline = try? device.makeRenderPipelineState(descriptor: copyPipelineDescriptor)
                
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
                
                // Rasterization scratch buffers
                finalTsBuffer = device.makeBuffer(length: maxPixels * MemoryLayout<Float>.stride, options: .storageModePrivate)
                finalIndexBuffer = device.makeBuffer(length: maxPixels * MemoryLayout<Int32>.stride, options: .storageModePrivate)
    }
    
    // MARK: - Data Loading
    func addPoints(positions: [SIMD3<Float>], colors: [SIMD3<Float>]) {
        let defaultScale = SIMD3<Float>(0.01, 0.01, 0.01)
        let covariances = Array(repeating: defaultScale, count: positions.count)
        addPoints(positions: positions, colors: colors, covariances: covariances)
    }

    func addPoints(positions: [SIMD3<Float>], colors: [SIMD3<Float>], covariances: [SIMD3<Float>]) {
        // 1. Append new data to meansBuffer, colorsBuffer
        // 2. Initialize scales (e.g., 0.01), quats (Identity), opacities (1.0)
        // 3. Resize intermediate buffers if maxPoints exceeded
        guard positions.count == colors.count && positions.count == covariances.count else {
            print("[GaussianSplatRenderer] Error: positions, colors, and covariances arrays must have the same count")
            return
        }
        
        let newPointCount = positions.count
        guard newPointCount > 0 else { 
            print("[GaussianSplatRenderer] addPoints called with 0 points")
            return 
        }
        
        print("[GaussianSplatRenderer] addPoints called with \(newPointCount) points (current total: \(pointCount))")
        
        // Get current point count from buffer capacity
        let currentCapacity = meansBuffer.length / MemoryLayout<SIMD3<Float>>.stride
        let device = meansBuffer.device
        
        // Calculate new total point count
        let totalPointCount = pointCount + newPointCount
        
        // Resize buffers if needed
        if totalPointCount > currentCapacity {
            let newCapacity = max(totalPointCount, currentCapacity * 2)
            print("[GaussianSplatRenderer] Resizing buffers from \(currentCapacity) to \(newCapacity)")
            
            // Save old buffers to copy data
            let oldMeansBuffer = meansBuffer
            let oldScalesBuffer = scalesBuffer
            let oldQuatsBuffer = quatsBuffer
            let oldOpacitiesBuffer = opacitiesBuffer
            let oldColorsBuffer = colorsBuffer
            
            // Reallocate Gaussian data buffers
            meansBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            scalesBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            quatsBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD4<Float>>.stride, options: .storageModeShared)
            opacitiesBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<Float>.stride, options: .storageModeShared)
            colorsBuffer = device.makeBuffer(length: newCapacity * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            
            // Copy existing data
            if pointCount > 0, let oldMeans = oldMeansBuffer, let oldScales = oldScalesBuffer,
               let oldQuats = oldQuatsBuffer, let oldOpacities = oldOpacitiesBuffer, let oldColors = oldColorsBuffer {
                memcpy(meansBuffer.contents(), oldMeans.contents(), pointCount * MemoryLayout<SIMD3<Float>>.stride)
                memcpy(scalesBuffer.contents(), oldScales.contents(), pointCount * MemoryLayout<SIMD3<Float>>.stride)
                memcpy(quatsBuffer.contents(), oldQuats.contents(), pointCount * MemoryLayout<SIMD4<Float>>.stride)
                memcpy(opacitiesBuffer.contents(), oldOpacities.contents(), pointCount * MemoryLayout<Float>.stride)
                memcpy(colorsBuffer.contents(), oldColors.contents(), pointCount * MemoryLayout<SIMD3<Float>>.stride)
            }
            
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
        
        // Append new position data (starting from current pointCount)
        let meansPointer = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: totalPointCount)
        for i in 0..<newPointCount {
            meansPointer[pointCount + i] = positions[i]
        }
        
        // Append new color data
        let colorsPointer = colorsBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: totalPointCount)
        for i in 0..<newPointCount {
            colorsPointer[pointCount + i] = colors[i]
        }
        
        // Append covariance data (scale components)
        let scalesPointer = scalesBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: totalPointCount)
        for i in 0..<newPointCount {
            scalesPointer[pointCount + i] = covariances[i]
        }
        
        // Initialize quaternions to identity rotation
        let quatsPointer = quatsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: totalPointCount)
        for i in 0..<newPointCount {
            quatsPointer[pointCount + i] = SIMD4<Float>(0, 0, 0, 1) // Identity quaternion (x, y, z, w)
        }
        
        // Initialize opacities to fully opaque
        let opacitiesPointer = opacitiesBuffer.contents().bindMemory(to: Float.self, capacity: totalPointCount)
        for i in 0..<newPointCount {
            opacitiesPointer[pointCount + i] = 1.0
        }
        
        // Update total point count
        self.pointCount = totalPointCount
        print("[GaussianSplatRenderer] Total points now: \(pointCount)")
    }
    
    // MARK: - Public Accessors
    
    /// Returns the current number of Gaussian splats
    func getPointCount() -> Int {
        return pointCount
    }
    
    /// Clears all points
    func clearPoints() {
        pointCount = 0
        print("[GaussianSplatRenderer] Cleared all points")
    }
    
    
    // MARK: - Draw Loop
    func draw(commandBuffer: MTLCommandBuffer, 
              viewMatrix: matrix_float4x4, 
              projectionMatrix: matrix_float4x4, 
              viewport: CGSize, 
              outputTexture: MTLTexture) {
        
        // Skip rendering if no points
        guard pointCount > 0 else {
            // Only print occasionally to avoid spam
            print("[GaussianSplatRenderer] No points to render")
            return
        }
        
        print("[GaussianSplatRenderer] Drawing \(pointCount) splats, viewport: \(viewport)")
        
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
        dispatchSort(commandBuffer: commandBuffer)
        
        // 5. Tile Range Pass
        // ------------------
        // Identifies the start and end indices for each tile in the sorted list.
        // Kernel: get_tile_bin_edges_kernel
        // Writes to: tileBinsBuffer
        dispatchTileBinning(commandBuffer: commandBuffer)
        
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
    
    // Ensure intermediate texture exists and matches size
    private func ensureIntermediateTexture(width: Int, height: Int) {
        if intermediateTexture == nil || 
           intermediateTexture!.width != width || 
           intermediateTexture!.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: width,
                height: height,
                mipmapped: false
            )
            descriptor.usage = [.shaderRead, .shaderWrite]
            descriptor.storageMode = .private
            intermediateTexture = device.makeTexture(descriptor: descriptor)
        }
    }
    
    private func dispatchDisplay(commandBuffer: MTLCommandBuffer, texture: MTLTexture) {
        // Ensure we have an intermediate texture of the right size
        ensureIntermediateTexture(width: texture.width, height: texture.height)
        
        guard let intermediateTex = intermediateTexture else { return }
        
        // First, write to the intermediate texture (which supports compute writes)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Display Pass"
        encoder.setComputePipelineState(displayPipeline)
        
        encoder.setBuffer(imgBuffer, offset: 0, index: 0)
        encoder.setTexture(intermediateTex, index: 0)
        
        var imgSize = SIMD2<UInt32>(UInt32(texture.width), UInt32(texture.height))
        encoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        
        let gridSize = MTLSize(width: texture.width, height: texture.height, depth: 1)
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
        
        // Now copy from intermediate texture to drawable using a render pass
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .dontCare
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        
        guard let copyPipeline = copyPipeline,
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
        
        renderEncoder.label = "Copy to Drawable"
        renderEncoder.setRenderPipelineState(copyPipeline)
        renderEncoder.setFragmentTexture(intermediateTex, index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renderEncoder.endEncoding()
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
        
        // Use pre-allocated scratch buffers for final_Ts and final_index
        encoder.setBuffer(finalTsBuffer, offset: 0, index: 9)
        encoder.setBuffer(finalIndexBuffer, offset: 0, index: 10)
        encoder.setBuffer(imgBuffer, offset: 0, index: 11)
        encoder.setBytes(&background, length: MemoryLayout<SIMD3<Float>>.size, index: 12)
        
        // Threadgroup config - use var so we can pass by reference
        var blockDim = SIMD2<UInt32>(16, 16)
        encoder.setBytes(&blockDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 13)
        
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(width: Int(tileBounds.x), height: Int(tileBounds.y), depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        
        encoder.endEncoding()
    }
}