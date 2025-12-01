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
import CoreImage.CIFilterBuiltins

/// Minimal headless runner for PromptDA + LoG probability
final class PromptDAEngine {
    struct Output {
        let depthPB: CVPixelBuffer            // 1C 32F, shape = outW×outH (model output)
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

        print("🚀 PromptDAEngine Initialization: \(bundleModelName)")

        // Load .mlmodelc/.mlpackage (try both extensions)
        var url: URL?
        
        // Try .mlmodelc first (compiled model)
        if let compiledURL = Bundle.main.url(forResource: bundleModelName, withExtension: "mlmodelc") {
            url = compiledURL
        }
        // Try .mlpackage (uncompiled model)
        else if let packageURL = Bundle.main.url(forResource: bundleModelName, withExtension: "mlpackage") {
            url = packageURL
        }
        // Try without extension (in case it's already included)
        else if let noExtURL = Bundle.main.url(forResource: bundleModelName, withExtension: nil) {
            url = noExtURL
        }
        
        guard let modelURL = url else {
            throw NSError(domain: "PromptDAEngine", code: 1, 
                         userInfo: [NSLocalizedDescriptionKey: "Model \(bundleModelName) not found in bundle."])
        }
        
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all  // Use ANE + GPU + CPU
        
        let model: MLModel
        do {
            model = try MLModel(contentsOf: modelURL, configuration: cfg)
            print("✅ CoreML model loaded successfully!")
        } catch {
            print("❌ Failed to load CoreML model: \(error.localizedDescription)")
            throw error
        }
        
        // Read expected promptDepth shape from model
        let expected = expectedPromptHW(from: model)
        
        // Use model's truth, fallback to manual promptHW
        let finalPromptHW = expected ?? promptHW

        // Reusable ARGB input for color
        var pb: CVPixelBuffer?
        let rgbStatus = CVPixelBufferCreate(kCFAllocatorDefault, Int(rgbSize.width), Int(rgbSize.height),
                            kCVPixelFormatType_32ARGB, nil, &pb)
        guard rgbStatus == kCVReturnSuccess, let rgbPB = pb else {
            throw NSError(domain: "PromptDAEngine", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create RGB buffer"])
        }

        // Reusable 1C float buf used when building LoG etc.
        var tmpBuf: CVPixelBuffer?
        let tmpStatus = CVPixelBufferCreate(kCFAllocatorDefault, Int(rgbSize.width), Int(rgbSize.height),
                            kCVPixelFormatType_OneComponent32Float, nil, &tmpBuf)
        guard tmpStatus == kCVReturnSuccess, let tmp1F = tmpBuf else {
            throw NSError(domain: "PromptDAEngine", code: 3, userInfo: [NSLocalizedDescriptionKey: "Failed to create temp buffer"])
        }

        // Initialize Metal compute pipeline for LoG smoothing
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw NSError(domain: "PromptDAEngine", code: 4, userInfo: [NSLocalizedDescriptionKey: "Metal is not available"])
        }

        guard let metalCommandQueue = metalDevice.makeCommandQueue() else {
            throw NSError(domain: "PromptDAEngine", code: 5, userInfo: [NSLocalizedDescriptionKey: "Failed to create Metal command queue"])
        }

        guard let library = metalDevice.makeDefaultLibrary() else {
            throw NSError(domain: "PromptDAEngine", code: 6, userInfo: [NSLocalizedDescriptionKey: "Failed to load Metal shader library"])
        }

        guard let smoothingFunction = library.makeFunction(name: "smoothLaplacianResponse") else {
            throw NSError(domain: "PromptDAEngine", code: 7, userInfo: [NSLocalizedDescriptionKey: "Failed to find smoothLaplacianResponse function"])
        }

        let smoothingPipeline: MTLComputePipelineState
        do {
            smoothingPipeline = try metalDevice.makeComputePipelineState(function: smoothingFunction)
        } catch {
            throw error
        }

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

        return .init(depthPB: depthPB)
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

    /// Converts an MLMultiArray (output from CoreML) into a CVPixelBuffer (needed for Metal textures).
    /// This is necessary because CoreML outputs multi-dimensional arrays (e.g. [1, 1, H, W]),
    /// but Metal textures require a standard 2D pixel buffer format (e.g. kCVPixelFormatType_OneComponent32Float).
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

    func makeNewLoGProbability(from rgbPB: CVPixelBuffer, size: CGSize) throws -> CVPixelBuffer {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create CIImage from the YCbCr pixel buffer
        let ciImage = CIImage(cvPixelBuffer: rgbPB)
        
        print("      → Input image extent: \(ciImage.extent)")
        
        // Convert to grayscale using luminance
        let gray = ciImage.toLuma()
        
        // Apply edge detection using CIEdges filter
        let edges = convolution3X3(inputImage: gray)!
        
        print("      → After CIEdges extent: \(edges.extent)")
        
        // Apply Gaussian blur to get broader response around edges
        let blurred = edges.applyingFilter("CIGaussianBlur", parameters: [
            "inputRadius": 1.0
        ])
        
        // Zero out edges (2 pixel border on each side)
        let border: CGFloat = 2
        let innerRect = CGRect(x: border,   
                               y: border, 
                               width: ciImage.extent.width - 2 * border, 
                               height: ciImage.extent.height - 2 * border)
        let maskedBlur = blurred.cropped(to: innerRect)
            .composited(over: CIImage(color: .black).cropped(to: ciImage.extent))
        
        // Clamp to [0, 1]
        let clamped = maskedBlur.applyingFilter("CIColorClamp", parameters: [
            "inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0),
            "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)
        ])
        
        // Convert to pixel buffer
        let resultPB = clamped.toPixelBuffer(context: ctx, pixelFormat: kCVPixelFormatType_OneComponent32Float, size: ciImage.extent.size)
        
        // Debug: Check final statistics
        CVPixelBufferLockBaseAddress(resultPB, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(resultPB, .readOnly) }
        
        if let ptr = CVPixelBufferGetBaseAddress(resultPB)?.assumingMemoryBound(to: Float.self) {
            var stats = (min: Float.infinity, max: -Float.infinity, sum: Float(0), nonZero: 0)
            let width = CVPixelBufferGetWidth(resultPB)
            let height = CVPixelBufferGetHeight(resultPB)
            let stride = CVPixelBufferGetBytesPerRow(resultPB) / MemoryLayout<Float>.stride
            let sampleSize = min(1000, width * height)
            
            for i in 0..<sampleSize {
                let y = i / width
                let x = i % width
                let val = ptr[y * stride + x]
                stats.min = min(stats.min, val)
                stats.max = max(stats.max, val)
                stats.sum += val
                if val > 0.001 { stats.nonZero += 1 }
            }
            print("      → Edge probability stats: min=\(stats.min), max=\(stats.max), avg=\(stats.sum/Float(sampleSize)), nonZero=\(stats.nonZero)")
        }
        
        let logTime = CFAbsoluteTimeGetCurrent() - startTime
        print("      → LoG generation time: \(String(format: "%.3f", logTime))s")
        
        return resultPB
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
        applyingFilter("CIColorMatrix", parameters: [
            "inputRVector": CIVector(x: 1, y: 0,      z: 0,      w: 0),
            "inputGVector": CIVector(x: 0,      y: 1, z: 0,      w: 0),
            "inputBVector": CIVector(x: 0,      y: 0,      z: 1, w: 0),
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
    
    func bicubicScaleTo(_ size: CGSize) -> CIImage {
        let scaleX = size.width / extent.width
        let scaleY = size.height / extent.height
        
        // Use bicubic scale transform for precise resizing without aspect ratio constraints
        let filter = CIFilter(name: "CIBicubicScaleTransform")!
        filter.setValue(self, forKey: kCIInputImageKey)
        filter.setValue(scaleX, forKey: "inputScale")
        filter.setValue(1.0, forKey: "inputAspectRatio")  // No aspect ratio adjustment
        filter.setValue(0.0, forKey: "inputB")  // Standard bicubic parameters
        filter.setValue(0.75, forKey: "inputC")
        
        // If aspect ratios differ, apply separate Y scaling
        if abs(scaleX - scaleY) > 0.001 {
            let intermediate = filter.outputImage!
            return intermediate.transformed(by: CGAffineTransform(scaleX: 1.0, y: scaleY / scaleX))
                .cropped(to: CGRect(origin: .zero, size: size))
        }
        
        return (filter.outputImage ?? self).cropped(to: CGRect(origin: .zero, size: size))
    }
    
    func toPixelBuffer(context: CIContext, pixelFormat: OSType, size: CGSize) -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault,
                                        Int(size.width),
                                        Int(size.height),
                                        pixelFormat,
                                        attrs as CFDictionary,
                                        &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            fatalError("Failed to create pixel buffer with status: \(status)")
        }
        
        context.render(self, to: buffer)
        return buffer
    }
}

func convolution3X3(inputImage: CIImage) -> CIImage? {
    let convolutionFilter = CIFilter.convolution3X3()
    convolutionFilter.inputImage = inputImage
    let kernel = CIVector(values: [
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
    ], count: 9)
    convolutionFilter.weights = kernel
    convolutionFilter.bias = 0.0
    return convolutionFilter.outputImage!
}

private extension CVPixelBuffer {
    var size: CGSize { .init(width: CVPixelBufferGetWidth(self),
                             height: CVPixelBufferGetHeight(self)) }
}
