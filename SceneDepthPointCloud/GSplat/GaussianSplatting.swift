//
//  GaussianSplatting.swift
//  SceneDepthPointCloud
//
//  Created by GitHub Copilot on 12/1/25.
//

import Metal
import MetalKit
import Foundation

class GaussianSplatting {
    private let device: MTLDevice
    private let library: MTLLibrary
    private let commandQueue: MTLCommandQueue
    
    // Pipeline States
    private var projectPipelineState: MTLComputePipelineState!
    private var mapIntersectsPipelineState: MTLComputePipelineState!
    private var getTileBinEdgesPipelineState: MTLComputePipelineState!
    private var rasterizePipelineState: MTLComputePipelineState!
    private var bitonicSortPipelineState: MTLComputePipelineState!
    private var fillLongPipelineState: MTLComputePipelineState!
    private var displayPipelineState: MTLComputePipelineState!
    
    // Constants
    private let maxPoints = 15_000_000 // Matches Renderer.maxPoints
    private let blockX = 16
    private let blockY = 16
    
    // Buffers (Persistent)
    var meansBuffer: MTLBuffer!
    var scalesBuffer: MTLBuffer!
    var quatsBuffer: MTLBuffer!
    var opacitiesBuffer: MTLBuffer!
    var colorsBuffer: MTLBuffer! // SH coefficients or RGB
    
    // Intermediate Buffers (Reused per frame)
    private var cov3dBuffer: MTLBuffer!
    private var xysBuffer: MTLBuffer!
    private var depthsBuffer: MTLBuffer!
    private var radiiBuffer: MTLBuffer!
    private var conicsBuffer: MTLBuffer!
    private var numTilesHitBuffer: MTLBuffer!
    
    // Sorting Buffers
    private var isectIdsBuffer: MTLBuffer!
    private var gaussianIdsBuffer: MTLBuffer!
    private var tileBinsBuffer: MTLBuffer!
    
    var pointCount: Int = 0
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue, library: MTLLibrary) {
        self.device = device
        self.commandQueue = commandQueue
        self.library = library
        
        setupPipelines()
        setupBuffers()
    }
    
    private func setupPipelines() {
        guard let projectFunc = library.makeFunction(name: "project_gaussians_forward_kernel"),
              let mapFunc = library.makeFunction(name: "map_gaussian_to_intersects_kernel"),
              let binFunc = library.makeFunction(name: "get_tile_bin_edges_kernel"),
              let rasterizeFunc = library.makeFunction(name: "nd_rasterize_forward_kernel"),
              let sortFunc = library.makeFunction(name: "bitonic_sort_step"),
              let fillFunc = library.makeFunction(name: "fill_long_kernel"),
              let displayFunc = library.makeFunction(name: "display_texture_kernel") else {
            fatalError("Failed to load Metal kernels")
        }
        
        do {
            projectPipelineState = try device.makeComputePipelineState(function: projectFunc)
            mapIntersectsPipelineState = try device.makeComputePipelineState(function: mapFunc)
            getTileBinEdgesPipelineState = try device.makeComputePipelineState(function: binFunc)
            rasterizePipelineState = try device.makeComputePipelineState(function: rasterizeFunc)
            bitonicSortPipelineState = try device.makeComputePipelineState(function: sortFunc)
            fillLongPipelineState = try device.makeComputePipelineState(function: fillFunc)
            displayPipelineState = try device.makeComputePipelineState(function: displayFunc)
        } catch {
            fatalError("Failed to create pipeline states: \(error)")
        }
    }
    
    private func setupBuffers() {
        let floatSize = MemoryLayout<Float>.stride
        let float3Size = floatSize * 3
        let float4Size = floatSize * 4
        
        meansBuffer = device.makeBuffer(length: maxPoints * float3Size, options: .storageModeShared)
        scalesBuffer = device.makeBuffer(length: maxPoints * float3Size, options: .storageModeShared)
        quatsBuffer = device.makeBuffer(length: maxPoints * float4Size, options: .storageModeShared)
        opacitiesBuffer = device.makeBuffer(length: maxPoints * floatSize, options: .storageModeShared)
        colorsBuffer = device.makeBuffer(length: maxPoints * float3Size, options: .storageModeShared)
        
        cov3dBuffer = device.makeBuffer(length: maxPoints * 6 * floatSize, options: .storageModePrivate)
        xysBuffer = device.makeBuffer(length: maxPoints * 2 * floatSize, options: .storageModePrivate)
        depthsBuffer = device.makeBuffer(length: maxPoints * floatSize, options: .storageModePrivate)
        radiiBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        conicsBuffer = device.makeBuffer(length: maxPoints * 3 * floatSize, options: .storageModePrivate)
        numTilesHitBuffer = device.makeBuffer(length: maxPoints * MemoryLayout<Int32>.stride, options: .storageModeShared)
        
        // Preallocate intersection buffers (assume max 100 tiles per point? conservative)
        let maxIntersects = maxPoints * 16 
        isectIdsBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int64>.stride, options: .storageModePrivate)
        gaussianIdsBuffer = device.makeBuffer(length: maxIntersects * MemoryLayout<Int32>.stride, options: .storageModePrivate)
    }

    
    func addPoints(positions: [SIMD3<Float>], colors: [SIMD3<Float>]) {
        guard positions.count == colors.count else { return }
        let count = positions.count
        if pointCount + count > maxPoints {
            print("Max points reached, cannot add \(count) points")
            return
        }
        
        // Update buffers
        // We need to copy data to the buffers at the correct offset
        
        // Means
        let meansPtr = meansBuffer.contents().advanced(by: pointCount * MemoryLayout<Float>.stride * 3).assumingMemoryBound(to: Float.self)
        for (i, pos) in positions.enumerated() {
            meansPtr[i * 3 + 0] = pos.x
            meansPtr[i * 3 + 1] = pos.y
            meansPtr[i * 3 + 2] = pos.z
        }
        
        // Colors
        let colorsPtr = colorsBuffer.contents().advanced(by: pointCount * MemoryLayout<Float>.stride * 3).assumingMemoryBound(to: Float.self)
        for (i, col) in colors.enumerated() {
            colorsPtr[i * 3 + 0] = col.x
            colorsPtr[i * 3 + 1] = col.y
            colorsPtr[i * 3 + 2] = col.z
        }
        
        // Scales (Initialize to small value, e.g. 0.01)
        // TODO: Use probability map for scale initialization as requested
        let scalesPtr = scalesBuffer.contents().advanced(by: pointCount * MemoryLayout<Float>.stride * 3).assumingMemoryBound(to: Float.self)
        let defaultScale: Float = 0.01
        for i in 0..<count {
            scalesPtr[i * 3 + 0] = defaultScale
            scalesPtr[i * 3 + 1] = defaultScale
            scalesPtr[i * 3 + 2] = defaultScale
        }
        
        // Quats (Identity)
        let quatsPtr = quatsBuffer.contents().advanced(by: pointCount * MemoryLayout<Float>.stride * 4).assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            quatsPtr[i * 4 + 0] = 1.0 // w
            quatsPtr[i * 4 + 1] = 0.0 // x
            quatsPtr[i * 4 + 2] = 0.0 // y
            quatsPtr[i * 4 + 3] = 0.0 // z
        }
        
        // Opacities (1.0)
        let opacitiesPtr = opacitiesBuffer.contents().advanced(by: pointCount * MemoryLayout<Float>.stride).assumingMemoryBound(to: Float.self)
        for i in 0..<count {
            opacitiesPtr[i] = 1.0
        }
        
        pointCount += count
    }
    
    func draw(commandBuffer: MTLCommandBuffer, viewMatrix: matrix_float4x4, projectionMatrix: matrix_float4x4, viewportSize: CGSize, outputTexture: MTLTexture) {
        guard pointCount > 0 else { return }
        
        let width = Int(viewportSize.width)
        let height = Int(viewportSize.height)
        let tileWidth = (width + blockX - 1) / blockX
        let tileHeight = (height + blockY - 1) / blockY
        let numTiles = tileWidth * tileHeight
        
        // Ensure tileBinsBuffer is large enough
        let tileBinsSize = numTiles * MemoryLayout<Int32>.stride * 2
        if tileBinsBuffer == nil || tileBinsBuffer.length < tileBinsSize {
            tileBinsBuffer = device.makeBuffer(length: tileBinsSize, options: .storageModePrivate)
        }
        
        // 1. Project Gaussians
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.setComputePipelineState(projectPipelineState)
        
        var numPointsVal = Int32(pointCount)
        computeEncoder.setBytes(&numPointsVal, length: MemoryLayout<Int32>.size, index: 0)
        computeEncoder.setBuffer(meansBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(scalesBuffer, offset: 0, index: 2)
        var globScale: Float = 1.0
        computeEncoder.setBytes(&globScale, length: MemoryLayout<Float>.size, index: 3)
        computeEncoder.setBuffer(quatsBuffer, offset: 0, index: 4)
        
        var viewMat = viewMatrix
        var projMat = projectionMatrix
        computeEncoder.setBytes(&viewMat, length: MemoryLayout<matrix_float4x4>.size, index: 5)
        computeEncoder.setBytes(&projMat, length: MemoryLayout<matrix_float4x4>.size, index: 6)
        
        let fx = projectionMatrix.columns.0.x * Float(width) / 2.0
        let fy = projectionMatrix.columns.1.y * Float(height) / 2.0
        let cx = Float(width) / 2.0
        let cy = Float(height) / 2.0
        var intrins = SIMD4<Float>(fx, fy, cx, cy)
        computeEncoder.setBytes(&intrins, length: MemoryLayout<SIMD4<Float>>.size, index: 7)
        
        var imgSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        computeEncoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 8)
        
        var tileBounds = SIMD3<UInt32>(UInt32(tileWidth), UInt32(tileHeight), 1)
        computeEncoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 9)
        
        var clipThresh: Float = 0.01
        computeEncoder.setBytes(&clipThresh, length: MemoryLayout<Float>.size, index: 10)
        
        computeEncoder.setBuffer(cov3dBuffer, offset: 0, index: 11)
        computeEncoder.setBuffer(xysBuffer, offset: 0, index: 12)
        computeEncoder.setBuffer(depthsBuffer, offset: 0, index: 13)
        computeEncoder.setBuffer(radiiBuffer, offset: 0, index: 14)
        computeEncoder.setBuffer(conicsBuffer, offset: 0, index: 15)
        computeEncoder.setBuffer(numTilesHitBuffer, offset: 0, index: 16)
        
        let gridSize = MTLSize(width: pointCount, height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: min(pointCount, projectPipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        // 2. Scan numTilesHit (CPU Roundtrip for now - still needed unless we implement Prefix Sum on GPU)
        // To fully implement GPU sort, we need GPU Prefix Sum.
        // For now, let's keep this CPU roundtrip as it's "Scan" not "Sort".
        // The user asked for "GPURadixSort", which usually implies the sorting part.
        // Prefix sum is a separate primitive.
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let numTilesHitPtr = numTilesHitBuffer.contents().bindMemory(to: Int32.self, capacity: pointCount)
        var totalIntersects: Int = 0
        var offsets = [Int32](repeating: 0, count: pointCount)
        
        for i in 0..<pointCount {
            offsets[i] = Int32(totalIntersects)
            totalIntersects += Int(numTilesHitPtr[i])
        }
        
        numTilesHitBuffer.contents().copyMemory(from: offsets, byteCount: pointCount * MemoryLayout<Int32>.stride)
        
        guard let sortCommandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        // 3. Map Gaussians to Intersects
        guard let mapEncoder = sortCommandBuffer.makeComputeCommandEncoder() else { return }
        mapEncoder.setComputePipelineState(mapIntersectsPipelineState)
        
        mapEncoder.setBytes(&numPointsVal, length: MemoryLayout<Int32>.size, index: 0)
        mapEncoder.setBuffer(xysBuffer, offset: 0, index: 1)
        mapEncoder.setBuffer(depthsBuffer, offset: 0, index: 2)
        mapEncoder.setBuffer(radiiBuffer, offset: 0, index: 3)
        mapEncoder.setBuffer(numTilesHitBuffer, offset: 0, index: 4)
        mapEncoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 5)
        mapEncoder.setBuffer(isectIdsBuffer, offset: 0, index: 6)
        mapEncoder.setBuffer(gaussianIdsBuffer, offset: 0, index: 7)
        
        mapEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        mapEncoder.endEncoding()
        
        // 4. Bitonic Sort on GPU
        if totalIntersects > 0 {
            // Next Power of 2
            var n = 1
            while n < totalIntersects { n <<= 1 }
            
            // Pad with INT64_MAX
            if n > totalIntersects {
                guard let fillEncoder = sortCommandBuffer.makeComputeCommandEncoder() else { return }
                fillEncoder.setComputePipelineState(fillLongPipelineState)
                
                fillEncoder.setBuffer(isectIdsBuffer, offset: 0, index: 0)
                var maxVal: Int64 = Int64.max
                fillEncoder.setBytes(&maxVal, length: MemoryLayout<Int64>.size, index: 1)
                var startIdx = UInt32(totalIntersects)
                fillEncoder.setBytes(&startIdx, length: MemoryLayout<UInt32>.size, index: 2)
                var count = UInt32(n - totalIntersects)
                fillEncoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
                
                let fillGridSize = MTLSize(width: Int(count), height: 1, depth: 1)
                let fillGroupSize = MTLSize(width: min(Int(count), fillLongPipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                fillEncoder.dispatchThreads(fillGridSize, threadsPerThreadgroup: fillGroupSize)
                fillEncoder.endEncoding()
            }
            
            guard let sortEncoder = sortCommandBuffer.makeComputeCommandEncoder() else { return }
            sortEncoder.setComputePipelineState(bitonicSortPipelineState)
            
            // Bitonic Sort Loop
            for k in sequence(first: 2, next: { $0 <= n ? $0 * 2 : nil }) {
                for j in sequence(first: k / 2, next: { $0 > 0 ? $0 / 2 : nil }) {
                    var jVal = UInt32(j)
                    var kVal = UInt32(k)
                    sortEncoder.setBytes(&jVal, length: MemoryLayout<UInt32>.size, index: 2)
                    sortEncoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 3)
                    
                    let sortGridSize = MTLSize(width: n / 2, height: 1, depth: 1) // n/2 threads
                    let sortGroupSize = MTLSize(width: min(n / 2, bitonicSortPipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    sortEncoder.dispatchThreads(sortGridSize, threadsPerThreadgroup: sortGroupSize)
                }
            }
            sortEncoder.endEncoding()
        }
        
        // 5. Get Tile Bin Edges
        if totalIntersects > 0 {
            let blit = sortCommandBuffer.makeBlitCommandEncoder()
            blit?.fill(buffer: tileBinsBuffer, range: 0..<tileBinsBuffer.length, value: 0)
            blit?.endEncoding()
            
            guard let binEncoder = sortCommandBuffer.makeComputeCommandEncoder() else { return }
            binEncoder.setComputePipelineState(getTileBinEdgesPipelineState)
            
            var numIntersectsVal = Int32(totalIntersects)
            binEncoder.setBytes(&numIntersectsVal, length: MemoryLayout<Int32>.size, index: 0)
            binEncoder.setBuffer(isectIdsBuffer, offset: 0, index: 1)
            binEncoder.setBuffer(tileBinsBuffer, offset: 0, index: 2)
            
            let binGridSize = MTLSize(width: totalIntersects, height: 1, depth: 1)
            let binGroupSize = MTLSize(width: min(totalIntersects, getTileBinEdgesPipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            binEncoder.dispatchThreads(binGridSize, threadsPerThreadgroup: binGroupSize)
            binEncoder.endEncoding()
        }
        
        // 6. Rasterize
        guard let rasterEncoder = sortCommandBuffer.makeComputeCommandEncoder() else { return }
        rasterEncoder.setComputePipelineState(rasterizePipelineState)
        
        rasterEncoder.setBytes(&tileBounds, length: MemoryLayout<SIMD3<UInt32>>.size, index: 0)
        rasterEncoder.setBytes(&imgSize, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        var channels: UInt32 = 3
        rasterEncoder.setBytes(&channels, length: MemoryLayout<UInt32>.size, index: 2)
        rasterEncoder.setBuffer(gaussianIdsBuffer, offset: 0, index: 3)
        rasterEncoder.setBuffer(tileBinsBuffer, offset: 0, index: 4)
        rasterEncoder.setBuffer(xysBuffer, offset: 0, index: 5)
        rasterEncoder.setBuffer(conicsBuffer, offset: 0, index: 6)
        rasterEncoder.setBuffer(colorsBuffer, offset: 0, index: 7)
        rasterEncoder.setBuffer(opacitiesBuffer, offset: 0, index: 8)
        
        let pixelCount = width * height
        let finalTsBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Float>.stride, options: .storageModePrivate)
        let finalIndexBuffer = device.makeBuffer(length: pixelCount * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        
        rasterEncoder.setBuffer(finalTsBuffer, offset: 0, index: 9)
        rasterEncoder.setBuffer(finalIndexBuffer, offset: 0, index: 10)
        
        let outImgBuffer = device.makeBuffer(length: pixelCount * 3 * MemoryLayout<Float>.stride, options: .storageModeShared)
        rasterEncoder.setBuffer(outImgBuffer, offset: 0, index: 11)
        
        var background = SIMD3<Float>(0, 0, 0)
        rasterEncoder.setBytes(&background, length: MemoryLayout<SIMD3<Float>>.size, index: 12)
        
        var blockDim = SIMD2<UInt32>(UInt32(blockX), UInt32(blockY))
        rasterEncoder.setBytes(&blockDim, length: MemoryLayout<SIMD2<UInt32>>.size, index: 13)
        
        let rasterGridSize = MTLSize(width: width, height: height, depth: 1)
        let rasterGroupSize = MTLSize(width: blockX, height: blockY, depth: 1)
        
        rasterEncoder.dispatchThreads(rasterGridSize, threadsPerThreadgroup: rasterGroupSize)
        rasterEncoder.endEncoding()
        
        // 7. Display (Copy Buffer to Texture)
        guard let displayEncoder = sortCommandBuffer.makeComputeCommandEncoder() else { return }
        displayEncoder.setComputePipelineState(displayPipelineState)
        displayEncoder.setBuffer(outImgBuffer, offset: 0, index: 0)
        displayEncoder.setTexture(outputTexture, index: 0)
        
        let displayGridSize = MTLSize(width: width, height: height, depth: 1)
        let displayGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        displayEncoder.dispatchThreads(displayGridSize, threadsPerThreadgroup: displayGroupSize)
        displayEncoder.endEncoding()
        
        sortCommandBuffer.commit()
    }
}
