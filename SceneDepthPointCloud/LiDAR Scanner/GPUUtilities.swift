import Foundation
import Metal
import MetalPerformanceShaders

/**
 GPU utility operations using MetalPerformanceShaders and custom kernels

 Provides:
 - Prefix sum (exclusive scan)
 - Radix sort on int64 keys
 - Gather (reorder array based on indices)
 */
class GPUUtilities {

    private let device: MTLDevice
    private let library: MTLLibrary
    private let commandQueue: MTLCommandQueue

    // Custom compute kernels
    private let prefixSumPipeline: MTLComputePipelineState
    private let gatherInt32Pipeline: MTLComputePipelineState
    private let gatherInt64Pipeline: MTLComputePipelineState
    private let initializeIndicesPipeline: MTLComputePipelineState

    // Radix sort kernels
    private let localHistogramPipeline: MTLComputePipelineState
    private let prefixSumRadixPipeline: MTLComputePipelineState
    private let reorderPipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.device = device

        guard let library = device.makeDefaultLibrary() else {
            throw NSError(domain: "GPUUtilities", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to load Metal library"])
        }
        self.library = library

        guard let queue = device.makeCommandQueue() else {
            throw NSError(domain: "GPUUtilities", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create command queue"])
        }
        self.commandQueue = queue

        // Create custom kernel pipelines
        self.prefixSumPipeline = try Self.makePipeline(device: device, library: library, functionName: "exclusivePrefixSumInt32")
        self.gatherInt32Pipeline = try Self.makePipeline(device: device, library: library, functionName: "gatherInt32")
        self.gatherInt64Pipeline = try Self.makePipeline(device: device, library: library, functionName: "gatherInt64")
        self.initializeIndicesPipeline = try Self.makePipeline(device: device, library: library, functionName: "initializeIndices")

        // Radix sort kernel pipelines
        self.localHistogramPipeline = try Self.makePipeline(device: device, library: library, functionName: "localHistogram")
        self.prefixSumRadixPipeline = try Self.makePipeline(device: device, library: library, functionName: "prefixSum")
        self.reorderPipeline = try Self.makePipeline(device: device, library: library, functionName: "reorder")
    }

    private static func makePipeline(device: MTLDevice, library: MTLLibrary, functionName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw NSError(domain: "GPUUtilities", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to find function: \(functionName)"])
        }
        return try device.makeComputePipelineState(function: function)
    }

    // MARK: - Prefix Sum (Exclusive Scan)

    /**
     Compute exclusive prefix sum on Int32 array

     Example: [1, 2, 3, 4] -> [0, 1, 3, 6]

     - Parameters:
       - commandBuffer: Command buffer to encode into
       - input: Source buffer with Int32 values
       - output: Destination buffer for prefix sum (can be same as input)
       - count: Number of elements
     - Returns: Total sum of all elements
     */
    func exclusivePrefixSum(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) -> Int {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return 0 }
        encoder.label = "Exclusive Prefix Sum"
        encoder.setComputePipelineState(prefixSumPipeline)

        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var countVar = UInt32(count)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 2)

        // For simplicity, use single-threaded approach for now
        // TODO: Implement parallel scan for better performance
        let threadgroupSize = MTLSize(width: min(count, 256), height: 1, depth: 1)
        let threadgroups = MTLSize(width: 1, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        // TODO: Return actual sum by reading from GPU or maintaining on GPU
        return 0  // Placeholder
    }

    /**
     Synchronously compute prefix sum and read result to CPU
     Useful for getting total sum (e.g., total intersection count)
     */
    func exclusivePrefixSumSync(input: MTLBuffer, output: MTLBuffer, count: Int) -> Int {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return 0 }

        _ = exclusivePrefixSum(commandBuffer: commandBuffer, input: input, output: output, count: count)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Read the last element of output + last element of input to get total
        let outputPtr = output.contents().bindMemory(to: Int32.self, capacity: count)
        let inputPtr = input.contents().bindMemory(to: Int32.self, capacity: count)

        if count > 0 {
            return Int(outputPtr[count - 1] + inputPtr[count - 1])
        }
        return 0
    }

    // MARK: - Radix Sort

    /**
     Sort int64 keys using custom Metal radix sort

     Implements 4-pass 8-bit radix sort (for 64-bit keys, we'd need 8 passes,
     but for prototype we'll do simpler 32-bit sort on lower bits)

     - Parameters:
       - commandBuffer: Command buffer to encode into
       - keys: Source buffer with int64 keys (we'll treat as int32 for simplicity)
       - keysSorted: Destination buffer for sorted keys
       - count: Number of elements
     */
    func radixSortInt64(
        commandBuffer: MTLCommandBuffer,
        keys: MTLBuffer,
        keysSorted: MTLBuffer,
        count: Int
    ) {
        // For simplicity in the prototype, just use a basic sort approach
        // A full radix sort would require 8 passes for 64-bit keys
        // For now, use a simpler approach: just copy the data
        // This allows the pipeline to compile and run

        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { return }
        blitEncoder.copy(from: keys, sourceOffset: 0, to: keysSorted, destinationOffset: 0, size: count * MemoryLayout<Int64>.stride)
        blitEncoder.endEncoding()

        // NOTE: Rendering order will be incorrect without proper sorting
        // But the system won't crash and you can still test other components

        // TODO: Implement full 8-pass radix sort for int64:
        // 1. For each of 8 passes (one per byte):
        //    a. Histogram: count occurrences of each digit value (0-255)
        //    b. Prefix sum: compute cumulative offsets
        //    c. Reorder: scatter elements to sorted positions
        // 2. Ping-pong between input/output buffers
    }

    /**
     Sort int64 keys and produce sorted indices

     This is useful for reordering other arrays based on sort order

     - Parameters:
       - commandBuffer: Command buffer to encode into
       - keys: Source buffer with int64 keys
       - keysSorted: Destination buffer for sorted keys
       - indices: Buffer to store sorted indices (will be initialized to [0,1,2,...] then sorted)
       - count: Number of elements
     */
    func radixSortInt64WithIndices(
        commandBuffer: MTLCommandBuffer,
        keys: MTLBuffer,
        keysSorted: MTLBuffer,
        indices: MTLBuffer,
        count: Int
    ) {
        // Step 1: Initialize indices to [0, 1, 2, ..., count-1]
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Initialize Indices"
        encoder.setComputePipelineState(initializeIndicesPipeline)

        encoder.setBuffer(indices, offset: 0, index: 0)
        var countVar = UInt32(count)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 1)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        // Step 2: Sort keys (and eventually track permutation)
        // For now, just sort the keys without tracking indices
        // TODO: Implement proper key-value radix sort to also permute indices
        radixSortInt64(commandBuffer: commandBuffer, keys: keys, keysSorted: keysSorted, count: count)

        // NOTE: Indices buffer is initialized but not permuted
        // This means gather will use identity permutation (no reordering)
        // Rendering will be incorrect until proper sort is implemented
    }

    // MARK: - Gather (Reorder)

    /**
     Reorder Int32 array based on indices

     output[i] = source[indices[i]]

     - Parameters:
       - commandBuffer: Command buffer to encode into
       - indices: Array of source indices (uint32)
       - source: Source data (int32)
       - destination: Output data (int32)
       - count: Number of elements to gather
     */
    func gatherInt32(
        commandBuffer: MTLCommandBuffer,
        indices: MTLBuffer,
        source: MTLBuffer,
        destination: MTLBuffer,
        count: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Gather Int32"
        encoder.setComputePipelineState(gatherInt32Pipeline)

        encoder.setBuffer(indices, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var countVar = UInt32(count)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    /**
     Reorder Int64 array based on indices
     */
    func gatherInt64(
        commandBuffer: MTLCommandBuffer,
        indices: MTLBuffer,
        source: MTLBuffer,
        destination: MTLBuffer,
        count: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Gather Int64"
        encoder.setComputePipelineState(gatherInt64Pipeline)

        encoder.setBuffer(indices, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
        encoder.setBuffer(destination, offset: 0, index: 2)
        var countVar = UInt32(count)
        encoder.setBytes(&countVar, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
}
