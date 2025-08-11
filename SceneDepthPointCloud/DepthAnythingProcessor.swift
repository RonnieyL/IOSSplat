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
        if let modelURL = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlpackage", subdirectory: "Models") ??
                          Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlpackage"),
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
        
        if let rawDepthTexture = runVisionInference(on: preprocessedImage, with: visionModel) {
            // Apply INRIA-style normalization for better point cloud quality
            return applyINRIANormalization(rawDepthTexture)
        }
        return nil
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
    
    // MARK: - INRIA-Style Depth Normalization (computationally expensive but accurate)
    private func applyINRIANormalization(_ depthTexture: MTLTexture) -> MTLTexture? {
        // Step 1: Calculate median and MAD (Median Absolute Deviation) like INRIA
        guard let statistics = calculateDepthStatistics(depthTexture) else {
            return normalizeDepthTexture(depthTexture)  // Fallback to simple normalization
        }
        
        // Step 2: Apply INRIA normalization: (depth - median) / MAD
        return applyStatisticalNormalization(depthTexture, median: statistics.median, mad: statistics.mad)
    }
    
    private struct DepthStatistics {
        let median: Float
        let mad: Float
    }
    
    private func calculateDepthStatistics(_ depthTexture: MTLTexture) -> DepthStatistics? {
        // Read texture data to CPU for statistical analysis
        let width = depthTexture.width
        let height = depthTexture.height
        let bytesPerRow = width * MemoryLayout<Float>.stride
        let totalBytes = height * bytesPerRow
        
        var depthData = [Float](repeating: 0, count: width * height)
        let region = MTLRegionMake2D(0, 0, width, height)
        
        depthTexture.getBytes(&depthData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)
        
        // Filter out invalid depths (0 or very small values)
        let validDepths = depthData.filter { $0 > 0.001 }
        guard !validDepths.isEmpty else { return nil }
        
        // Calculate median (INRIA's robust center measure)
        let sortedDepths = validDepths.sorted()
        let median = sortedDepths.count % 2 == 0 ?
            (sortedDepths[sortedDepths.count/2 - 1] + sortedDepths[sortedDepths.count/2]) / 2 :
            sortedDepths[sortedDepths.count/2]
        
        // Calculate MAD (Median Absolute Deviation - INRIA's robust scale measure)
        let absoluteDeviations = validDepths.map { abs($0 - median) }
        let sortedDeviations = absoluteDeviations.sorted()
        let mad = sortedDeviations.count % 2 == 0 ?
            (sortedDeviations[sortedDeviations.count/2 - 1] + sortedDeviations[sortedDeviations.count/2]) / 2 :
            sortedDeviations[sortedDeviations.count/2]
        
        // Prevent division by zero
        let safeMad = max(mad, 0.001)
        
        print("📊 Depth Statistics - Median: \(median), MAD: \(safeMad)")
        
        return DepthStatistics(median: median, mad: safeMad)
    }
    
    private func applyStatisticalNormalization(_ depthTexture: MTLTexture, median: Float, mad: Float) -> MTLTexture? {
        // Create compute shader for INRIA-style normalization
        guard let library = metalDevice.makeDefaultLibrary(),
              let function = library.makeFunction(name: "inriaNormalizeDepth"),
              let pipelineState = try? metalDevice.makeComputePipelineState(function: function) else {
            print("⚠️ Could not create INRIA normalization pipeline, using simple normalization")
            return normalizeDepthTexture(depthTexture)
        }
        
        // Create output texture
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: depthTexture.pixelFormat,
            width: depthTexture.width,
            height: depthTexture.height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let normalizedTexture = metalDevice.makeTexture(descriptor: descriptor),
              let commandBuffer = metalDevice.makeCommandQueue()?.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return depthTexture
        }
        
        // Pass statistics to shader
        var params = SIMD4<Float>(median, mad, 0, 0)
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(normalizedTexture, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (depthTexture.width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (depthTexture.height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return normalizedTexture
    }
    
    // MARK: - Fallback Simple Normalization
    private func normalizeDepthTexture(_ depthTexture: MTLTexture) -> MTLTexture? {
        // Create compute shader for depth normalization
        guard let library = metalDevice.makeDefaultLibrary(),
              let function = library.makeFunction(name: "normalizeDepth"),
              let pipelineState = try? metalDevice.makeComputePipelineState(function: function) else {
            print("⚠️ Could not create depth normalization pipeline, using raw depth")
            return depthTexture
        }
        
        // Create output texture
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: depthTexture.pixelFormat,
            width: depthTexture.width,
            height: depthTexture.height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let normalizedTexture = metalDevice.makeTexture(descriptor: descriptor),
              let commandBuffer = metalDevice.makeCommandQueue()?.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return depthTexture
        }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(depthTexture, index: 0)
        encoder.setTexture(normalizedTexture, index: 1)
        
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (depthTexture.width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (depthTexture.height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return normalizedTexture
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
