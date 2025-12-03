import Metal

class PrefixSum {
    private let device: MTLDevice
    private var pipelinePartial: MTLComputePipelineState!
    private var pipelineScan: MTLComputePipelineState!
    private var pipelineFinal: MTLComputePipelineState!
    
    private var partialSumsBuffer: MTLBuffer?
    private var constStep1Buffer: MTLBuffer?
    private var constStep2Buffer: MTLBuffer?
    private var constStep3Buffer: MTLBuffer?
    
    private var numElements: Int = 0
    private var numPartialSums: Int = 0
    
    private let numThreadsPerThreadgroup = 512
    
    private struct Constants {
        var num_elements: UInt32
        var num_threads_per_partial_sum: UInt32
    }
    
    init(device: MTLDevice) {
        self.device = device
        let library = device.makeDefaultLibrary()!
        
        pipelinePartial = try! device.makeComputePipelineState(function: library.makeFunction(name: "mg_get_partial_sums_32_X_int")!)
        pipelineScan = try! device.makeComputePipelineState(function: library.makeFunction(name: "mg_scan_threadgroupwise_32_X_int")!)
        pipelineFinal = try! device.makeComputePipelineState(function: library.makeFunction(name: "mg_scan_final_32_X_int")!)
    }
    
    func compute(commandBuffer: MTLCommandBuffer, inputBuffer: MTLBuffer, outputBuffer: MTLBuffer, count: Int) {
        if count != numElements {
            configure(count: count)
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Prefix Sum"
        
        // Step 1: Partial Sums
        encoder.setComputePipelineState(pipelinePartial)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(partialSumsBuffer, offset: 0, index: 1)
        encoder.setBuffer(constStep1Buffer, offset: 0, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: numPartialSums, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: numThreadsPerThreadgroup, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
        
        // Step 2: Scan Partial Sums
        encoder.setComputePipelineState(pipelineScan)
        encoder.setBuffer(partialSumsBuffer, offset: 0, index: 0)
        encoder.setBuffer(constStep2Buffer, offset: 0, index: 1)
        let threadsStep2 = (numPartialSums + 31) / 32 * 32
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsStep2, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
        
        // Step 3: Final Add
        encoder.setComputePipelineState(pipelineFinal)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(partialSumsBuffer, offset: 0, index: 2)
        encoder.setBuffer(constStep3Buffer, offset: 0, index: 3)
        encoder.dispatchThreadgroups(MTLSize(width: numPartialSums, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: numThreadsPerThreadgroup, height: 1, depth: 1))
        
        encoder.endEncoding()
    }
    
    private func configure(count: Int) {
        self.numElements = count
        
        // Configuration logic ported from C++
        let requestedPartialSums = min(numThreadsPerThreadgroup, 1024)
        
        // roundup_X(x, n) = ((n + x - 1) / x) * x
        let n = (count + requestedPartialSums - 1) / requestedPartialSums
        let x = numThreadsPerThreadgroup
        let numThreadsPerPartialSum = ((n + x - 1) / x) * x
        
        self.numPartialSums = (count + numThreadsPerPartialSum - 1) / numThreadsPerPartialSum
        
        // Reallocate buffers
        partialSumsBuffer = device.makeBuffer(length: numPartialSums * MemoryLayout<Int32>.stride, options: .storageModePrivate)
        
        var c1 = Constants(num_elements: UInt32(count), num_threads_per_partial_sum: UInt32(numThreadsPerPartialSum))
        constStep1Buffer = device.makeBuffer(bytes: &c1, length: MemoryLayout<Constants>.stride, options: .storageModeShared)
        
        var c2 = Constants(num_elements: UInt32(numPartialSums), num_threads_per_partial_sum: 0)
        constStep2Buffer = device.makeBuffer(bytes: &c2, length: MemoryLayout<Constants>.stride, options: .storageModeShared)
        
        var c3 = Constants(num_elements: UInt32(count), num_threads_per_partial_sum: UInt32(numThreadsPerPartialSum))
        constStep3Buffer = device.makeBuffer(bytes: &c3, length: MemoryLayout<Constants>.stride, options: .storageModeShared)
    }
}
