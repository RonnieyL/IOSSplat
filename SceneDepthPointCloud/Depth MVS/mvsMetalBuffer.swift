/*
See LICENSE folder for this sample's licensing information.

Abstract:
Type-safe utility for working with MTLBuffers.
*/

import MetalKit

protocol Resource {
    associatedtype Element
}

/// A wrapper around MTLBuffer which provides type safe access and assignment to the underlying MTLBuffer's contents.

struct MetalBuffer<Element>: Resource {
        
    /// The underlying MTLBuffer.
    fileprivate let buffer: MTLBuffer
    
    /// The index that the buffer should be bound to during encoding.
    /// Should correspond with the index that the buffer is expected to be at in Metal shaders.
    fileprivate let index: Int
    
    /// The number of elements of T the buffer can hold.
    let count: Int
    var stride: Int {
        MemoryLayout<Element>.stride
    }

    /// Initializes the buffer with zeros, the buffer holds `count` elements of type Element.
    init(device: MTLDevice, count: Int, index: UInt32, label: String? = nil, options: MTLResourceOptions = []) {
        
        guard let buffer = device.makeBuffer(length: MemoryLayout<Element>.stride * count, options: options) else {
            fatalError("Failed to create MTLBuffer.")
        }
        self.buffer = buffer
        self.buffer.label = label
        self.count = count
        self.index = Int(index)
    }
    
    /// Initializes the buffer with the contents of `array`.
    init(device: MTLDevice, array: [Element], index: UInt32, options: MTLResourceOptions = [])  {
        
        guard let buffer = device.makeBuffer(bytes: array, length: MemoryLayout<Element>.stride * array.count, options: .storageModeShared) else {
            fatalError("Failed to create MTLBuffer")
        }
        self.buffer = buffer
        self.count = array.count
        self.index = Int(index)
    }
    
    /// Replaces the buffer's memory at `index` with `element`.
    subscript(index: Int) -> Element {
        get {
            precondition(index < count, "Index \(index) is greater than the buffer length (\(count))")
            return buffer.contents().advanced(by: index * stride).assumingMemoryBound(to: Element.self).pointee
        }
        
        set {
            precondition(index < count, "Index \(index) is greater than the buffer length (\(count))")
            buffer.contents().advanced(by: index * stride).assumingMemoryBound(to: Element.self).pointee = newValue
        }
    }
    
    /// Replaces the buffer's memory with the elements of `array`.
    func assign<C: Collection>(with collection: C) where C.Element == Element {
        let byteCount = collection.count * stride
        precondition(collection.count <= count, "Collection \(collection.count) is greater than the buffer length (\(count))")
        collection.withContiguousStorageIfAvailable { bytes in
            buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: byteCount)
        }
    }
    
    /// Returns the buffer's memory as an array.
    func toArray() -> [Element] {
        return convertToArray(buffer: buffer, count: count)
    }

}

extension MetalBuffer: MTLBuffer {
    
    var contents: UnsafeMutableRawPointer {
        return buffer.contents()
    }
    
    func didModifyRange(_ range: Range<Int>) {
        buffer.didModifyRange(range)
    }
    
    var length: Int {
        return buffer.length
    }
    
    func makeAliasable() {
        return buffer.makeAliasable()
    }
    
    var isAliasable: Bool {
        return buffer.isAliasable
    }
    
    var device: MTLDevice {
        return buffer.device
    }
    
    var label: String? {
        get {
            return buffer.label
        }
        set {
            buffer.label = newValue
        }
    }
    
    var cpuCacheMode: MTLCPUCacheMode {
        return buffer.cpuCacheMode
    }
    
    var storageMode: MTLStorageMode {
        return buffer.storageMode
    }
    
    var resourceOptions: MTLResourceOptions {
        return buffer.resourceOptions
    }
    
    func setPurgeableState(_ state: MTLPurgeableState) -> MTLPurgeableState {
        return buffer.setPurgeableState(state)
    }
    
    var heap: MTLHeap? {
        return buffer.heap
    }
    
    var heapOffset: Int {
        return buffer.heapOffset
    }
    
    var allocatedSize: Int {
        return buffer.allocatedSize
    }
    
    func makeRemoteBufferView(_ device: MTLDevice) -> MTLBuffer? {
        return buffer.makeRemoteBufferView(device)
    }
    
    var remoteStorageBuffer: MTLBuffer? {
        return buffer.remoteStorageBuffer
    }
    
    func newRemoteBufferView(for device: MTLDevice) -> MTLBuffer? {
        return buffer.newRemoteBufferView(for: device)
    }
}

private func convertToArray<T>(buffer: MTLBuffer, count: Int) -> [T] {
    let pointer = buffer.contents().assumingMemoryBound(to: T.self)
    let buffPointer = UnsafeBufferPointer(start: pointer, count: count)
    return Array(buffPointer)
}
