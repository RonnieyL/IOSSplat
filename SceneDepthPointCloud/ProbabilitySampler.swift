//
//  ProbabilitySampler.swift
//  SceneDepthPointCloud
//
//  Draw samples from a 1C 32F probability map using Bernoulli trials
//

import Accelerate
import CoreVideo
import simd

struct SampledPixel {
    let x: Int
    let y: Int
    let p: Float   // effective probability used
}

/// Bernoulli sampling from probability maps - stochastic sample count based on scene complexity
enum ProbabilitySampler {
    
    /// New Bernoulli sampling approach - no fixed count, naturally adapts to scene complexity
    static func samplePixelsBernoulli(probPB: CVPixelBuffer, scale: Float = 1.0) -> [SIMD2<Int>] {
        let samples = samplePixels(from: probPB, scale: scale)
        
        // Convert SampledPixel array to SIMD2<Int> array for compatibility
        return samples.map { SIMD2<Int>($0.x, $0.y) }
    }
    
    /// Core Bernoulli sampling implementation
    static func samplePixels(from probBuffer: CVPixelBuffer, scale: Float = 1.0) -> [SampledPixel] {
        CVPixelBufferLockBaseAddress(probBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(probBuffer, .readOnly) }
        
        let width  = CVPixelBufferGetWidth(probBuffer)
        let height = CVPixelBufferGetHeight(probBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(probBuffer)
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(probBuffer) else {
            return []
        }
        
        print("   🎲 ProbabilitySampler (Bernoulli):")
        print("      • Probability map: \(width)×\(height)")
        print("      • Scale factor: \(scale)")
        
        // OneComponent32Float → 4 bytes per pixel
        let floatsPerRow = bytesPerRow / MemoryLayout<Float>.size
        let ptr = baseAddress.bindMemory(to: Float.self, capacity: floatsPerRow * height)
        
        var samples: [SampledPixel] = []
        samples.reserveCapacity(width * height / 10) // heuristic
        
        var totalProb: Float = 0
        var validPixels = 0
        
        for y in 0..<height {
            let rowPtr = ptr.advanced(by: y * floatsPerRow)
            for x in 0..<width {
                let rawP = rowPtr[x]
                
                // scale and clamp like init_proba * init_proba_scaler
                var p = rawP * scale
                if p <= 0 { continue }
                if p > 1 { p = 1 }   // avoid probs > 1
                
                totalProb += p
                validPixels += 1
                
                // flip a coin: U ~ Uniform(0,1)
                let u = Float.random(in: 0..<1)
                if u < p {
                    samples.append(SampledPixel(x: x, y: y, p: p))
                }
            }
        }
        
        print("      • Valid pixels: \(validPixels)/\(width*height)")
        print("      • Total probability mass: \(String(format: "%.3f", totalProb))")
        print("      • Sampled \(samples.count) points (stochastic)")
        print("      • Effective sampling rate: \(String(format: "%.1f%%", Float(samples.count) / Float(validPixels) * 100))")
        
        return samples
    }
    
    /// Legacy CDF-based sampling (kept for compatibility/fallback)
    static func samplePixels(probPB: CVPixelBuffer, count: Int, unique: Bool = true) -> [SIMD2<Int>] {
        CVPixelBufferLockBaseAddress(probPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(probPB, .readOnly) }

        let W = CVPixelBufferGetWidth(probPB)
        let H = CVPixelBufferGetHeight(probPB)
        let stride = CVPixelBufferGetBytesPerRow(probPB) / MemoryLayout<Float>.stride
        let src = CVPixelBufferGetBaseAddress(probPB)!.assumingMemoryBound(to: Float.self)

        print("   🎲 ProbabilitySampler (Legacy CDF):")
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
