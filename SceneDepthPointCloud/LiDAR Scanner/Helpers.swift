/*
See LICENSE folder for this sample’s licensing information.

Abstract:
General Helper methods and properties
*/

import ARKit

typealias Float2 = SIMD2<Float>
typealias Float3 = SIMD3<Float>

extension Float {
    static let degreesToRadian = Float.pi / 180
}

extension matrix_float3x3 {
    mutating func copy(from affine: CGAffineTransform) {
        columns.0 = Float3(Float(affine.a), Float(affine.c), Float(affine.tx))
        columns.1 = Float3(Float(affine.b), Float(affine.d), Float(affine.ty))
        columns.2 = Float3(0, 0, 1)
    }
}

extension matrix_float4x4 {
    @inlinable mutating func copy(from m: simd_float4x4) { self = m }
}
extension matrix_float3x3 {
    @inlinable mutating func copy(from m: simd_float3x3) { self = m }
    @inlinable mutating func copy(from m4: simd_float4x4) { self = m4.upperLeft3x3() }
}

extension simd_float4x4 {
    @inlinable func upperLeft3x3() -> simd_float3x3 {
        simd_float3x3(
            SIMD3<Float>(columns.0.x, columns.0.y, columns.0.z),
            SIMD3<Float>(columns.1.x, columns.1.y, columns.1.z),
            SIMD3<Float>(columns.2.x, columns.2.y, columns.2.z)
        )
    }
}

