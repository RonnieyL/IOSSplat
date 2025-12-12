//
//  ProbabilitySampler.swift
//  SceneDepthPointCloud
//
//  Draw N samples from a 1C 32F probability map (LoG-based)
//

import Accelerate
import CoreVideo
import simd

/// Draw N samples from a 1C 32F probability map PB (values sum to ~1).
/// Returns (x,y) pixels in image space.
enum ProbabilitySampler {
    static func samplePixels(probPB: CVPixelBuffer, count: Int, unique: Bool = true) -> [SIMD2<Int>] {
        CVPixelBufferLockBaseAddress(probPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(probPB, .readOnly) }

        let W = CVPixelBufferGetWidth(probPB)
        let H = CVPixelBufferGetHeight(probPB)
        let stride = CVPixelBufferGetBytesPerRow(probPB) / MemoryLayout<Float>.stride
        let src = CVPixelBufferGetBaseAddress(probPB)!.assumingMemoryBound(to: Float.self)

        print("   🎲 ProbabilitySampler:")
        print("      • Probability map: \(W)×\(H)")
        print("      • Requested samples: \(count), unique: \(unique)")

        // Build CDF respecting stride (row-major order)
        var cdf = [Float](repeating: 0, count: W*H)
        var acc: Float = 0
        var validPixels = 0
        
        for y in 0..<H {
            let row = y * stride
            for x in 0..<W {
                let prob = src[row + x]
                acc += prob
                cdf[y * W + x] = acc  // Store in row-major order (not stride-based)
                if prob > 0 { validPixels += 1 }
            }
        }
        
        let total = acc
        print("      • Probability sum: \(String(format: "%.6f", total)) (should be ~1.0)")
        print("      • Valid pixels: \(validPixels)/\(W*H)")
        
        guard total > 1e-6 else {
            print("      ⚠️ Warning: Probability sum near zero, falling back to uniform sampling")
            // Fallback: return uniformly distributed points
            var uniform = [SIMD2<Int>]()
            let step = max(1, (W * H) / count)
            for i in Swift.stride(from: 0, to: W * H, by: step) {
                if uniform.count >= count { break }
                let y = i / W
                let x = i % W
                uniform.append(.init(x, y))
            }
            return uniform
        }
        
        // Normalize CDF to [0, 1]
        if abs(total - 1.0) > 0.01 {
            print("      ⚠️ Warning: Probability sum is \(String(format: "%.3f", total)), renormalizing")
            let invTotal = 1.0 / total
            for i in 0..<cdf.count {
                cdf[i] *= invTotal
            }
        }

        var out = [SIMD2<Int>]()
        out.reserveCapacity(count)
        var used = Set<Int>()
        
        let maxAttempts = unique ? count * 3 : count
        var attempts = 0
        
        for _ in 0..<maxAttempts {
            if out.count >= count { break }
            attempts += 1
            
            // Generate random value in [0, 1)
            let u = Float.random(in: 0..<1)
            
            // Binary search for first CDF entry >= u
            var lo = 0
            var hi = cdf.count - 1
            
            while lo < hi {
                let mid = (lo + hi) >> 1
                if cdf[mid] < u {
                    lo = mid + 1
                } else {
                    hi = mid
                }
            }
            
            // Ensure we don't go out of bounds
            let idx = min(lo, cdf.count - 1)
            
            if unique && used.contains(idx) { continue }
            used.insert(idx)
            
            let y = idx / W
            let x = idx % W
            out.append(.init(x, y))
        }
        
        print("      • Sampled \(out.count) points in \(attempts) attempts")
        
        return out
    }
}
