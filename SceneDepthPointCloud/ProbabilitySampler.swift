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

        // Flatten → CDF
        var cdf = [Float](repeating: 0, count: W*H)
        var acc: Float = 0
        for y in 0..<H {
            let row = y*stride
            for x in 0..<W {
                acc += src[row + x]
                cdf[y*W + x] = acc
            }
        }
        let total = max(acc, 1e-6)
        print("      • Probability sum: \(String(format: "%.6f", total)) (should be ~1.0)")
        
        let invTotal = 1.0 / total
        vDSP_vsmul(cdf, 1, [invTotal], &cdf, 1, vDSP_Length(cdf.count))

        var out = [SIMD2<Int>]()
        out.reserveCapacity(count)
        var used = Set<Int>()
        
        let maxAttempts = unique ? count * 3 : count  // Allow multiple attempts if unique
        var attempts = 0
        
        for _ in 0..<maxAttempts {
            if out.count >= count { break }
            attempts += 1
            
            let u = Float.random(in: 0..<1)
            // lower_bound on CDF
            var lo = 0, hi = cdf.count-1
            while lo < hi {
                let mid = (lo + hi) >> 1
                if cdf[mid] < u { lo = mid + 1 } else { hi = mid }
            }
            if unique && used.contains(lo) { continue }
            used.insert(lo)
            let y = lo / W, x = lo % W
            out.append(.init(x, y))
        }
        
        print("      • Sampled \(out.count) points in \(attempts) attempts")
        
        return out
    }
}
