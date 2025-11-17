//
//  GSplatStruct.swift
//  SceneDepthPointCloud
//
//  CPU-side Gaussian Splat representation
//

import Foundation
import simd

final class GaussianSplat {
    var position: simd_float3      // 3D world position (12 bytes)
    var color: simd_float3         // RGB color 0-255 (12 bytes)
    var opacity: Float             // Alpha/transparency 0-1 (4 bytes)
    var scale: simd_float3         // 3D scale for ellipsoid axes (12 bytes)
    var rotation: simd_quatf       // Quaternion rotation (16 bytes)

    // Total: 56 bytes per Gaussian

    init(position: simd_float3,
         color: simd_float3,
         opacity: Float,
         scale: simd_float3 = simd_float3(0.01, 0.01, 0.01),
         rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)) {
        self.position = position
        self.color = color * 255  // Normalize to 0-255 range for export
        self.opacity = opacity
        self.scale = scale
        self.rotation = rotation
    }
}
