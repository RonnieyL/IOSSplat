import Metal

// Reuse the tested LiDAR buffer wrapper for MVS code.
typealias MVSMetalBuffer<T> = MetalBuffer<T>

// If you referenced MVSResource anywhere, map it too:
typealias MVSResource = Resource
