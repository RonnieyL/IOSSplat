//
//  PromptDAEngine.swift
//  SceneDepthPointCloud
//
//  Minimal headless runner for PromptDA + LoG probability
//

import CoreImage
import CoreML
import Accelerate
import AVFoundation
import Metal

/// Minimal headless runner for PromptDA + LoG probability
final class PromptDAEngine {
    struct Output {
        let depthPB: CVPixelBuffer            // 1C 32F, shape = outW×outH (model output)
        let probPB:  CVPixelBuffer            // 1C 32F, same size as depthPB (LoG response)
    }

    private let ctx = CIContext()
    private let model: MLModel
    private let rgbSize: CGSize
    private let promptHW: (h: Int, w: Int)?   // set if you want LiDAR prompt; else nil

    // Reusable buffers
    private var rgbPB: CVPixelBuffer
    private var tmp1F: CVPixelBuffer

    // Metal compute pipeline for LoG smoothing
    private let metalDevice: MTLDevice
    private let metalCommandQueue: MTLCommandQueue
    private let smoothingPipeline: MTLComputePipelineState

    // Private init - use static create() method instead
    private init(model: MLModel, rgbSize: CGSize, promptHW: (Int, Int)?, rgbPB: CVPixelBuffer, tmp1F: CVPixelBuffer, metalDevice: MTLDevice, metalCommandQueue: MTLCommandQueue, smoothingPipeline: MTLComputePipelineState) {
        self.model = model
        self.rgbSize = rgbSize
        self.promptHW = promptHW
        self.rgbPB = rgbPB
        self.tmp1F = tmp1F
        self.metalDevice = metalDevice
        self.metalCommandQueue = metalCommandQueue
        self.smoothingPipeline = smoothingPipeline
    }
    
    // Helper to read expected prompt H/W from model description
    private static func expectedPromptHW(from model: MLModel) -> (h: Int, w: Int)? {
        guard let d = model.modelDescription.inputDescriptionsByName["promptDepth"],
              let c = d.multiArrayConstraint else { return nil }
        // Shape is [N, C, H, W]
        let shape = c.shape.map { $0.intValue }
        guard shape.count == 4 else { return nil }
        return (shape[2], shape[3])
    }
    
    static func create(bundleModelName: String = "PromptDA_vits_518x518_prompt192x256",
                       rgbSize: CGSize = .init(width: 518, height: 518),
                       promptHW: (Int, Int)? = nil) throws -> PromptDAEngine {

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🚀 PromptDAEngine Initialization Starting...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("   • Model name: \(bundleModelName)")
        print("   • RGB input size: \(Int(rgbSize.width))×\(Int(rgbSize.height))")

        // Load .mlmodelc/.mlpackage (try both extensions)
        print("   • Searching for model in bundle...")
        
        var url: URL?
        var foundExtension: String?
        
        // Try .mlmodelc first (compiled model)
        if let compiledURL = Bundle.main.url(forResource: bundleModelName, withExtension: "mlmodelc") {
            url = compiledURL
            foundExtension = "mlmodelc"
            print("   • Found compiled model: \(bundleModelName).mlmodelc")
        }
        // Try .mlpackage (uncompiled model)
        else if let packageURL = Bundle.main.url(forResource: bundleModelName, withExtension: "mlpackage") {
            url = packageURL
            foundExtension = "mlpackage"
            print("   • Found model package: \(bundleModelName).mlpackage")
        }
        // Try without extension (in case it's already included)
        else if let noExtURL = Bundle.main.url(forResource: bundleModelName, withExtension: nil) {
            url = noExtURL
            foundExtension = "none"
            print("   • Found model without extension: \(bundleModelName)")
        }
        
        guard let modelURL = url else {
            print("❌ Model NOT found in bundle!")
            print("   • Searched for: \(bundleModelName).mlmodelc")
            print("   • Searched for: \(bundleModelName).mlpackage")
            print("   • Bundle path: \(Bundle.main.bundlePath)")
            
            // List available .mlmodel files in bundle
            if let resourcePath = Bundle.main.resourcePath {
                print("   • Scanning bundle for .mlmodel* files...")
                do {
                    let files = try FileManager.default.contentsOfDirectory(atPath: resourcePath)
                    let mlFiles = files.filter { $0.contains(".mlmodel") || $0.contains(".mlpackage") }
                    if mlFiles.isEmpty {
                        print("   • No ML models found in bundle")
                    } else {
                        print("   • Found ML files:")
                        mlFiles.forEach { print("     - \($0)") }
                    }
                } catch {
                    print("   • Could not scan bundle: \(error.localizedDescription)")
                }
            }
            
            throw NSError(domain: "PromptDAEngine", code: 1, 
                         userInfo: [NSLocalizedDescriptionKey: "Model \(bundleModelName) not found in bundle. Check that the .mlpackage or .mlmodelc file is added to the Xcode project target."])
        }
        
        print("   • Model file location: \(modelURL.path)")
        print("   • Loading model into CoreML...")
        
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all  // Use ANE + GPU + CPU
        print("   • Compute units: ANE + GPU + CPU")
        
        let model: MLModel
        do {
            model = try MLModel(contentsOf: modelURL, configuration: cfg)
            print("✅ CoreML model loaded successfully!")
            
            // Print model metadata
            let desc = model.modelDescription
            print("   • Model metadata:")
            print("     - Input features: \(desc.inputDescriptionsByName.keys.joined(separator: ", "))")
            print("     - Output features: \(desc.outputDescriptionsByName.keys.joined(separator: ", "))")
            
            // Print input shapes
            for (name, feature) in desc.inputDescriptionsByName {
                if let constraint = feature.multiArrayConstraint {
                    print("     - \(name) shape: \(constraint.shape)")
                } else if let constraint = feature.imageConstraint {
                    print("     - \(name) size: \(constraint.pixelsWide)×\(constraint.pixelsHigh)")
                }
            }
        } catch {
            print("❌ Failed to load CoreML model: \(error.localizedDescription)")
            throw error
        }
        
        // Read expected promptDepth shape from model
        let expected = expectedPromptHW(from: model)
        if let p = expected {
            print("   • Model expects promptDepth shape [1,1,\(p.h),\(p.w)] (NCHW)")
        } else {
            print("   • Model has no promptDepth or it's not constrained")
        }
        
        // Use model's truth, fallback to manual promptHW
        let finalPromptHW = expected ?? promptHW

        print("   • Creating reusable buffers...")
        
        // Reusable ARGB input for color
        var pb: CVPixelBuffer?
        let rgbStatus = CVPixelBufferCreate(kCFAllocatorDefault, Int(rgbSize.width), Int(rgbSize.height),
                            kCVPixelFormatType_32ARGB, nil, &pb)
        guard rgbStatus == kCVReturnSuccess, let rgbPB = pb else {
            print("❌ Failed to create RGB buffer: status=\(rgbStatus)")
            throw NSError(domain: "PromptDAEngine", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create RGB buffer"])
        }
        print("   • RGB buffer created: \(Int(rgbSize.width))×\(Int(rgbSize.height)) ARGB")

        // Reusable 1C float buf used when building LoG etc.
        var tmpBuf: CVPixelBuffer?
        let tmpStatus = CVPixelBufferCreate(kCFAllocatorDefault, Int(rgbSize.width), Int(rgbSize.height),
                            kCVPixelFormatType_OneComponent32Float, nil, &tmpBuf)
        guard tmpStatus == kCVReturnSuccess, let tmp1F = tmpBuf else {
            print("❌ Failed to create temp buffer: status=\(tmpStatus)")
            throw NSError(domain: "PromptDAEngine", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create temp buffer"])
        }
        print("   • Temp buffer created: \(Int(rgbSize.width))×\(Int(rgbSize.height)) Float32")

        // Initialize Metal compute pipeline for LoG smoothing
        print("   • Setting up Metal compute pipeline...")
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            print("❌ Failed to create Metal device")
            throw NSError(domain: "PromptDAEngine", code: 4, userInfo: [NSLocalizedDescriptionKey: "Metal is not available"])
        }
        print("   • Metal device: \(metalDevice.name)")

        guard let metalCommandQueue = metalDevice.makeCommandQueue() else {
            print("❌ Failed to create Metal command queue")
            throw NSError(domain: "PromptDAEngine", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command queue"])
        }

        guard let library = metalDevice.makeDefaultLibrary() else {
            print("❌ Failed to load Metal shader library")
            throw NSError(domain: "PromptDAEngine", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to load Metal shader library"])
        }

        guard let smoothingFunction = library.makeFunction(name: "smoothLaplacianResponse") else {
            print("❌ Failed to find smoothLaplacianResponse function in Metal library")
            throw NSError(domain: "PromptDAEngine", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to find smoothLaplacianResponse function"])
        }

        let smoothingPipeline: MTLComputePipelineState
        do {
            smoothingPipeline = try metalDevice.makeComputePipelineState(function: smoothingFunction)
            print("   • Metal compute pipeline created successfully")
        } catch {
            print("❌ Failed to create Metal compute pipeline: \(error.localizedDescription)")
            throw error
        }

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ PromptDAEngine initialization complete!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        return PromptDAEngine(model: model, rgbSize: rgbSize, promptHW: finalPromptHW, rgbPB: pb!, tmp1F: tmp1F, metalDevice: metalDevice, metalCommandQueue: metalCommandQueue, smoothingPipeline: smoothingPipeline)
    }

    /// Runs PromptDA. If `lidarPB` is provided and the model expects it, we pass it as prompt.
    func predict(rgbPB inPB: CVPixelBuffer, lidarPB: CVPixelBuffer?) throws -> Output {
        let inW = CVPixelBufferGetWidth(inPB)
        let inH = CVPixelBufferGetHeight(inPB)
        print("🔵 PromptDAEngine.predict():")
        print("   • Input RGB: \(inW)×\(inH)")
        
        // 1) Orient & resize RGB → model size
        let rgbCI = CIImage(cvPixelBuffer: inPB)
        let scaled = rgbCI.lanczosTo(rgbSize)
        print("   • Scaled RGB to model input: \(Int(rgbSize.width))×\(Int(rgbSize.height))")
        ctx.render(scaled, to: rgbPB)

        // 2) Optional: prepare LiDAR prompt to [1,1,H,W] Float16
        var prompt: MLMultiArray?
        if let (H, W) = promptHW, let lidarPB = lidarPB {
            let lidarW = CVPixelBufferGetWidth(lidarPB)
            let lidarH = CVPixelBufferGetHeight(lidarPB)
            print("   • Processing LiDAR prompt: \(lidarW)×\(lidarH) (width×height) → MLMultiArray[1,1,\(H),\(W)]")
            prompt = try makePromptArray(from: lidarPB, H: H, W: W)
        } else {
            print("   • No LiDAR prompt provided")
        }

        // 3) Run model
        let feats = try MLDictionaryFeatureProvider(dictionary: [
            "colorImage": MLFeatureValue(pixelBuffer: rgbPB),
            "promptDepth": prompt.map(MLFeatureValue.init(multiArray:))
        ].compactMapValues { $0 })

        print("   • Running CoreML inference...")
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        
        let out = try model.prediction(from: feats)
        
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart
        print("   • Inference completed in \(String(format: "%.3f", inferenceTime))s")
        
        let outName = model.modelDescription.outputDescriptionsByName.keys.first!
        let fv = out.featureValue(for: outName)!

        // Depth buffer from output (supports pb or multiarray)
        let depthPB: CVPixelBuffer =
            fv.imageBufferValue ??
            make1F(from: fv.multiArrayValue!)  // convert [1,1,H,W] → 1C 32F PB

        let outDepthW = CVPixelBufferGetWidth(depthPB)
        let outDepthH = CVPixelBufferGetHeight(depthPB)
        print("   • Model output depth: \(outDepthW)×\(outDepthH)")

        // 4) Build LoG probability on the RGB at the same resolution as depth
        print("   • Computing LoG probability map...")
        
        // Use the RENDERED pixel buffer (rgbPB), not the CIImage (scaled)
        // The CIImage may have wrong extent due to aspect ratio preservation
        let rgbForLoG = CIImage(cvPixelBuffer: rgbPB)
        let colorSpaceName = rgbForLoG.colorSpace?.name as String? ?? "nil"
        print("      → Input RGB to LoG: extent=\(rgbForLoG.extent), colorSpace=\(colorSpaceName)")
        
        let probPB = try makeLoGProbability(from: rgbForLoG, size: depthPB.size)
        let probW = CVPixelBufferGetWidth(probPB)
        let probH = CVPixelBufferGetHeight(probPB)
        print("   • LoG probability map: \(probW)×\(probH)")

        return .init(depthPB: depthPB, probPB: probPB)
    }

    // MARK: - Prompt & LoG helpers

    private func makePromptArray(from depthPB: CVPixelBuffer, H: Int, W: Int) throws -> MLMultiArray {
        // Create array [1,1,H,W] Float16
        let arr = try MLMultiArray(shape: [1, 1, H as NSNumber, W as NSNumber], dataType: .float16)

        let inW = CVPixelBufferGetWidth(depthPB)
        let inH = CVPixelBufferGetHeight(depthPB)
        print("      → makePromptArray: input \(inW)×\(inH) (W×H), expected [H,W]=[\(H),\(W)]")

        // Fast paths: direct copy or transpose. Else, resize to (W×H) first.
        var srcPB: CVPixelBuffer = depthPB

        if !(inW == W && inH == H) && !(inW == H && inH == W) {
            // Resize to match expected W×H
            var tmp: CVPixelBuffer?
            CVPixelBufferCreate(kCFAllocatorDefault, W, H, kCVPixelFormatType_OneComponent32Float, nil, &tmp)
            let ci = CIImage(cvPixelBuffer: depthPB)
            let resized = ci.lanczosTo(.init(width: W, height: H))
            ctx.render(resized, to: tmp!)
            srcPB = tmp!
            print("      → Resized depth to \(W)×\(H)")
        }

        CVPixelBufferLockBaseAddress(srcPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(srcPB, .readOnly) }

        let sW = CVPixelBufferGetWidth(srcPB)
        let sH = CVPixelBufferGetHeight(srcPB)
        let stride = CVPixelBufferGetBytesPerRow(srcPB) / MemoryLayout<Float>.stride
        let src = CVPixelBufferGetBaseAddress(srcPB)!.assumingMemoryBound(to: Float.self)

        var minDepth: Float = .infinity
        var maxDepth: Float = -.infinity
        var validCount = 0

        // Case A: already [W×H] → row-major copy into [H,W]
        if sW == W && sH == H {
            for y in 0..<H {
                let row = y * stride
                for x in 0..<W {
                    let v = src[row + x]
                    let val = v.isFinite ? v : 0
                    arr[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: val)
                    
                    if v.isFinite && v > 0 {
                        minDepth = min(minDepth, v)
                        maxDepth = max(maxDepth, v)
                        validCount += 1
                    }
                }
            }
            print("      → Copied row-major (no transpose)")
        }
        // Case B: it's [H×W] → transpose while copying into [H,W]
        else if sW == H && sH == W {
            for y in 0..<H {            // dest y
                for x in 0..<W {        // dest x
                    // source at (row=x, col=y) because src is H×W
                    let v = src[x * stride + y]
                    let val = v.isFinite ? v : 0
                    arr[[0, 0, y as NSNumber, x as NSNumber]] = NSNumber(value: val)
                    
                    if v.isFinite && v > 0 {
                        minDepth = min(minDepth, v)
                        maxDepth = max(maxDepth, v)
                        validCount += 1
                    }
                }
            }
            print("      → Transposed copy (input was H×W)")
        }

        print("      → Prompt depth stats: min=\(String(format: "%.3f", minDepth))m, max=\(String(format: "%.3f", maxDepth))m, valid=\(validCount)/\(H*W)")
        
        return arr
    }

    private func make1F(from arr: MLMultiArray) -> CVPixelBuffer {
        let h = arr.shape[arr.shape.count-2].intValue
        let w = arr.shape.last!.intValue
        
        // Create Metal-compatible pixel buffer with IOSurface backing
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, w, h, 
                                        kCVPixelFormatType_OneComponent32Float, 
                                        attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let pixelBuffer = pb else {
            print("      ❌ Failed to create pixel buffer in make1F: status=\(status)")
            fatalError("Failed to create pixel buffer from MLMultiArray")
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        
        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer) / MemoryLayout<Float>.stride
        let dst = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: Float.self)
        
        for y in 0..<h {
            for x in 0..<w {
                dst[y*stride + x] = arr[[0,0,y as NSNumber,x as NSNumber]].floatValue
            }
        }
        
        return pixelBuffer
    }

    /// LoG probability map: Laplacian → abs → clamp values to [0,1]
    private func makeLoGProbability(from rgb: CIImage, size: CGSize) throws -> CVPixelBuffer {
        print("      → makeLoGProbability: input extent \(rgb.extent), target size \(Int(size.width))×\(Int(size.height))")

        // Step 1: Grayscale (luma)
        let gray = rgb.toLuma()
        
        // Step 2: Laplacian kernel
        let laplacian = gray.applyingFilter("CIConvolution3X3", parameters: [
            "inputWeights": CIVector(values: [0, 1, 0, 1, -4, 1, 0, 1, 0], count: 9),
            "inputBias": 0
        ])
        
        // Step 3: Absolute value
        let abs = laplacian.applyingFilter("CIColorAbsoluteDifference", parameters: [
            "inputImage2": CIImage(color: .black)
        ])
        
        // Step 4: Clamp pixel VALUES to [0, 1] (not extent, just values)
        let clamped = abs.applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
        ])
        
        // Step 5: Render to pixel buffer
        let pixelBuffer = try clamped.toPixelBuffer(context: ctx, pixelFormat: kCVPixelFormatType_OneComponent32Float, size: size)
        
        // 6. Check what we got
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let W = Int(size.width)
        let H = Int(size.height)
        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer) / MemoryLayout<Float>.stride
        let buf = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: Float.self)
        
        var count = 0
        var sum: Float = 0
        var maxVal: Float = 0
        
        for y in 0..<H {
            for x in 0..<W {
                let v = buf[y * stride + x]
                if v > 0 {
                    count += 1
                    sum += v
                    maxVal = max(maxVal, v)
                }
            }
        }
        
        print("      → Rendered buffer: \(count)/\(W*H) non-zero, max=\(String(format: "%.6f", maxVal)), sum=\(String(format: "%.3f", sum))")
        
        return pixelBuffer
    }
}

private extension CIImage {
    func toLuma() -> CIImage {
        applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 0.299, y: 0,      z: 0,      w: 0),
            "inputGVector": CIVector(x: 0,      y: 0.587, z: 0,      w: 0),
            "inputBVector": CIVector(x: 0,      y: 0,      z: 0.114, w: 0),
            "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0)
        ])
    }
    
    func toGray() -> CIImage {
        // Equal weighting of R, G, B channels: output = (R + G + B) / 3
        // Each output channel gets the same gray value
        let w: CGFloat = 1.0 / 3.0
        return applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: w, y: w, z: w, w: 0),  // R channel gets (R+G+B)/3
            "inputGVector": CIVector(x: w, y: w, z: w, w: 0),  // G channel gets (R+G+B)/3
            "inputBVector": CIVector(x: w, y: w, z: w, w: 0),  // B channel gets (R+G+B)/3
            "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0)
        ])
    }
    
    func lanczosTo(_ size: CGSize) -> CIImage {
        let sx = size.width/extent.width, sy = size.height/extent.height
        let f = CIFilter(name:"CILanczosScaleTransform")!
        f.setValue(self,  forKey:kCIInputImageKey)
        f.setValue(sx,    forKey:"inputScale")
        f.setValue(sx/sy, forKey:"inputAspectRatio")
        let img = (f.outputImage ?? self).cropped(to: .init(origin:.zero, size:size))
        return img
    }
    
    func toPixelBuffer(context: CIContext, pixelFormat: OSType, size: CGSize) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                                        pixelFormat, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let pixelBuffer = pb else {
            throw NSError(domain: "PromptDAEngine", code: 3, 
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer: status=\(status)"])
        }
        
        // CRITICAL: Use explicit bounds starting at origin (0,0)
        // CIImage extent might have non-zero origin which causes misalignment
        let renderBounds = CGRect(origin: .zero, size: size)
        let colorSpace = CGColorSpaceCreateDeviceGray()  // Use device gray for single-channel
        
        context.render(self, to: pixelBuffer, bounds: renderBounds, colorSpace: colorSpace)
        return pixelBuffer
    }
}

private extension CVPixelBuffer {
    var size: CGSize { .init(width: CVPixelBufferGetWidth(self),
                             height: CVPixelBufferGetHeight(self)) }
}
