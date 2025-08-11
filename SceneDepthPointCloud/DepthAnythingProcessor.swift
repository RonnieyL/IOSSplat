//
//  DepthAnythingProcessor.swift
//  SceneDepthPointCloud
//
//  CoreML Depth-Anything-V2 integration for real-time depth estimation
//

import Foundation
import CoreML
import Vision
import CoreImage
import Metal
import MetalKit
import ARKit

class DepthAnythingProcessor {
    
    // MARK: - Properties
    private var depthModel: MLModel?
    private var visionModel: VNCoreMLModel?
    private let processingQueue = DispatchQueue(label: "depth-processing", qos: .userInitiated)
    private let ciContext: CIContext
    private let metalDevice: MTLDevice
    
    // Model input requirements (Depth-Anything-V2 Small)
    private let modelInputSize = CGSize(width: 518, height: 518)
    private let isModelLoaded: Bool
    
    // MARK: - Initialization
    init(metalDevice: MTLDevice) {
        self.metalDevice = metalDevice
        self.ciContext = CIContext(mtlDevice: metalDevice)
        
        // Try to load the CoreML model
        if let modelURL = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlpackage"),
           let model = try? MLModel(contentsOf: modelURL) {
            self.depthModel = model
            self.visionModel = try? VNCoreMLModel(for: model)
            self.isModelLoaded = true
            print("✅ Depth-Anything-V2 model loaded successfully")
        } else {
            print("⚠️ Depth-Anything-V2 model not found. MVS mode will use synthetic depth.")
            self.isModelLoaded = false
        }
    }
    
    // MARK: - Public Methods
    
    /// Process ARFrame to generate depth map using Depth-Anything-V2
    func processDepth(from frame: ARFrame, completion: @escaping (MTLTexture?) -> Void) {
        guard isModelLoaded else {
            // Fallback to synthetic depth if model not available
            completion(generateSyntheticDepthTexture(from: frame))
            return
        }
        
        processingQueue.async { [weak self] in
            guard let self = self else {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            
            let depthTexture = self.runDepthInference(on: frame)
            
            DispatchQueue.main.async {
                completion(depthTexture)
            }
        }
    }
    
    /// Synchronous depth processing for real-time use
    func processDepthSync(from frame: ARFrame) -> MTLTexture? {
        guard isModelLoaded else {
            return generateSyntheticDepthTexture(from: frame)
        }
        
        return runDepthInference(on: frame)
    }
    
    // MARK: - Private Methods
    
    private func runDepthInference(on frame: ARFrame) -> MTLTexture? {
        // Convert ARFrame to CIImage
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Preprocess image for model input
        guard let preprocessedImage = preprocessImage(ciImage) else {
            print("❌ Failed to preprocess image for depth inference")
            return nil
        }
        
        // Run inference using Vision framework (more efficient than direct CoreML)
        guard let visionModel = visionModel else {
            return runDirectCoreMLInference(on: preprocessedImage)
        }
        
        return runVisionInference(on: preprocessedImage, with: visionModel)
    }
    
    private func preprocessImage(_ image: CIImage) -> CIImage? {
        // Resize to model input size while maintaining aspect ratio
        let inputAspect = modelInputSize.width / modelInputSize.height
        let imageAspect = image.extent.width / image.extent.height
        
        var targetSize = modelInputSize
        if imageAspect > inputAspect {
            // Image is wider - fit to height
            targetSize.width = modelInputSize.height * imageAspect
        } else {
            // Image is taller - fit to width  
            targetSize.height = modelInputSize.width / imageAspect
        }
        
        // Scale the image
        let scaleX = targetSize.width / image.extent.width
        let scaleY = targetSize.height / image.extent.height
        let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        
        // Center crop to exact model input size
        let cropRect = CGRect(
            x: (scaledImage.extent.width - modelInputSize.width) / 2,
            y: (scaledImage.extent.height - modelInputSize.height) / 2,
            width: modelInputSize.width,
            height: modelInputSize.height
        )
        
        return scaledImage.cropped(to: cropRect)
    }
    
    private func runVisionInference(on image: CIImage, with visionModel: VNCoreMLModel) -> MTLTexture? {
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFit
        
        let handler = VNImageRequestHandler(ciImage: image, options: [:])
        
        do {
            try handler.perform([request])
            
            guard let results = request.results,
                  let pixelBufferResult = results.first as? VNPixelBufferObservation else {
                print("❌ No depth results from Vision inference")
                return nil
            }
            
            return convertDepthBufferToTexture(pixelBufferResult.pixelBuffer)
            
        } catch {
            print("❌ Vision inference failed: \(error)")
            return nil
        }
    }
    
    private func runDirectCoreMLInference(on image: CIImage) -> MTLTexture? {
        guard let depthModel = depthModel else { return nil }
        
        // Convert CIImage to CVPixelBuffer
        guard let pixelBuffer = convertCIImageToPixelBuffer(image) else {
            print("❌ Failed to convert CIImage to CVPixelBuffer")
            return nil
        }
        
        // Create MLFeatureProvider
        guard let featureProvider = try? MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]) else {
            print("❌ Failed to create feature provider")
            return nil
        }
        
        // Run prediction
        do {
            let prediction = try depthModel.prediction(from: featureProvider)
            
            // Extract depth output (model-specific - may need adjustment)
            if let depthOutput = prediction.featureValue(for: "depth")?.multiArrayValue {
                return convertMultiArrayToTexture(depthOutput)
            }
            
        } catch {
            print("❌ CoreML prediction failed: \(error)")
        }
        
        return nil
    }
    
    private func convertCIImageToPixelBuffer(_ image: CIImage) -> CVPixelBuffer? {
        let attributes: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(modelInputSize.width),
            Int(modelInputSize.height),
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        ciContext.render(image, to: buffer)
        return buffer
    }
    
    private func convertDepthBufferToTexture(_ pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = metalDevice.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        // Copy pixel buffer data to Metal texture
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let region = MTLRegionMake2D(0, 0, width, height)
        
        texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: baseAddress,
            bytesPerRow: bytesPerRow
        )
        
        return texture
    }
    
    private func convertMultiArrayToTexture(_ multiArray: MLMultiArray) -> MTLTexture? {
        guard multiArray.dataType == .float32 else {
            print("❌ Unsupported multi-array data type")
            return nil
        }
        
        let shape = multiArray.shape
        guard shape.count >= 2 else {
            print("❌ Invalid multi-array shape")
            return nil
        }
        
        let width = shape[shape.count - 1].intValue
        let height = shape[shape.count - 2].intValue
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = metalDevice.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        // Copy multi-array data to texture
        let floatPointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        let region = MTLRegionMake2D(0, 0, width, height)
        
        texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: floatPointer,
            bytesPerRow: width * MemoryLayout<Float>.stride
        )
        
        return texture
    }
    
    // MARK: - Fallback Synthetic Depth
    private func generateSyntheticDepthTexture(from frame: ARFrame) -> MTLTexture? {
        let width = Int(frame.camera.imageResolution.width / 4) // Lower resolution for performance
        let height = Int(frame.camera.imageResolution.height / 4)
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = metalDevice.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        // Generate synthetic depth data
        let depthData = (0..<(width * height)).map { i -> Float in
            let x = Float(i % width) / Float(width)
            let y = Float(i / width) / Float(height)
            return 1.0 + 0.5 * sin(x * 10) * cos(y * 10) // Synthetic wave pattern
        }
        
        let region = MTLRegionMake2D(0, 0, width, height)
        texture.replace(
            region: region,
            mipmapLevel: 0,
            withBytes: depthData,
            bytesPerRow: width * MemoryLayout<Float>.stride
        )
        
        return texture
    }
    
    // MARK: - Public Status
    var isReady: Bool {
        return isModelLoaded
    }
    
    var modelStatus: String {
        return isModelLoaded ? "Depth-Anything-V2 Ready" : "Using Synthetic Depth"
    }
}
