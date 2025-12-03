import Metal

class BitonicSort {
    private let device: MTLDevice
    private var sortPipeline: MTLComputePipelineState!
    private var padPipeline: MTLComputePipelineState!
    
    init(device: MTLDevice) {
        self.device = device
        let library = device.makeDefaultLibrary()!
        sortPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "bitonic_sort_kernel")!)
        padPipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "bitonic_pad_kernel")!)
    }
    
    func sort(commandBuffer: MTLCommandBuffer, keys: MTLBuffer, values: MTLBuffer, count: Int) {
        // count must be a power of 2
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Bitonic Sort"
        encoder.setComputePipelineState(sortPipeline)
        encoder.setBuffer(keys, offset: 0, index: 0)
        encoder.setBuffer(values, offset: 0, index: 1)
        
        var k: UInt32 = 2
        while k <= UInt32(count) {
            var j = k >> 1
            while j > 0 {
                encoder.setBytes(&j, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 3)
                
                let gridSize = MTLSize(width: count, height: 1, depth: 1)
                let threadGroupSize = MTLSize(width: min(sortPipeline.maxTotalThreadsPerThreadgroup, count), height: 1, depth: 1)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
                
                encoder.memoryBarrier(scope: .buffers)
                j >>= 1
            }
            k <<= 1
        }
        encoder.endEncoding()
    }
    
    func pad(commandBuffer: MTLCommandBuffer, keys: MTLBuffer, startIndex: Int, endIndex: Int) {
        guard startIndex < endIndex else { return }
        let count = endIndex - startIndex
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Bitonic Pad"
        encoder.setComputePipelineState(padPipeline)
        encoder.setBuffer(keys, offset: 0, index: 0)
        var start = UInt32(startIndex)
        var end = UInt32(endIndex)
        encoder.setBytes(&start, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBytes(&end, length: MemoryLayout<UInt32>.size, index: 2)
        
        let gridSize = MTLSize(width: count, height: 1, depth: 1)
        let threadGroupSize = MTLSize(width: min(padPipeline.maxTotalThreadsPerThreadgroup, count), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize)
        
        encoder.endEncoding()
    }
}
