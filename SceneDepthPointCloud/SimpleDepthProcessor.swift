//
//  SimpleDepthProcessor.swift
//  SceneDepthPointCloud
//
//  Created to fix depth extraction issues - Based on HuggingFace CoreML example
//

import Foundation
import ARKit
import CoreML
import Vision
import Metal
import CoreImage

class SimpleDepthProcessor {
    private let metalDevice: MTLDevice
    private var depthModel: MLModel?
    private var visionModel: VNCoreMLModel?
    private let ciContext: CIContext
    
    // Model configuration (matching HuggingFace Depth-Anything-V2-Small)
    private let modelInputSize = CGSize(width: 518, height: 518)
    
    init(metalDevice: MTLDevice) {
        self.metalDevice = metalDevice
        self.ciContext = CIContext(mtlDevice: metalDevice)
        loadDepthModel()
    }
    
    private func loadDepthModel() {
        // Try multiple possible locations for the model
        var modelURL: URL?
        
        // First try: Models subdirectory (most likely location)
        if let url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlpackage", subdirectory: "Models") {
            modelURL = url
            print("✅ Found model in Models subdirectory")
        }
        // Second try: Root bundle
        else if let url = Bundle.main.url(forResource: "DepthAnythingV2SmallF16", withExtension: "mlpackage") {
            modelURL = url
            print("✅ Found model in root bundle")
        }
        // Third try: Check all bundle resources
        else {
            print("❌ Could not find DepthAnythingV2SmallF16.mlpackage")
            print("📋 Available .mlpackage files in bundle:")
            if let resourcePath = Bundle.main.resourcePath {
                let enumerator = FileManager.default.enumerator(atPath: resourcePath)
                while let file = enumerator?.nextObject() as? String {
                    if file.hasSuffix(".mlpackage") {
                        print("   - \(file)")
                    }
                }
            }
            return
        }
        
        guard let finalURL = modelURL else {
            print("❌ No model URL found")
            return
        }
        
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU + CPU
            
            depthModel = try MLModel(contentsOf: finalURL, configuration: config)
            visionModel = try VNCoreMLModel(for: depthModel!)
            
            print("✅ Depth-Anything-V2 model loaded successfully from: \(finalURL.lastPathComponent)")
        } catch {
            print("❌ Failed to load depth model from \(finalURL): \(error)")
        }
    }
    
    /// Main depth processing function (simplified and robust)
    func processDepth(from frame: ARFrame) -> MTLTexture? {
        guard let visionModel = visionModel else {
            print("❌ Depth model not loaded")
            return nil
        }
        
        // Step 1: Convert ARFrame to CIImage
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Step 2: Resize for model input (HuggingFace approach)
        guard let resizedImage = resizeImageForModel(ciImage) else {
            print("❌ Failed to resize image")
            return nil
        }
        
        // Step 3: Run Vision inference (most reliable method)
        guard let depthTexture = runVisionInference(resizedImage, model: visionModel) else {
            print("❌ Vision inference failed")
            return nil
        }
        
        // Step 4: Simple normalization (no complex statistics for now)
        return normalizeDepthSimple(depthTexture)
    }
    
    private func resizeImageForModel(_ image: CIImage) -> CIImage? {
        // Simple center crop and resize (like HuggingFace example)
        let imageRect = image.extent
        let targetSize = modelInputSize
        
        // Calculate scale to fit while maintaining aspect ratio
        let scaleX = targetSize.width / imageRect.width
        let scaleY = targetSize.height / imageRect.height
        let scale = max(scaleX, scaleY)
        
        // Scale and center
        let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        
        // Center crop to exact model size
        let cropRect = CGRect(
            x: (scaledImage.extent.width - targetSize.width) / 2,
            y: (scaledImage.extent.height - targetSize.height) / 2,
            width: targetSize.width,
            height: targetSize.height
        )
        
        return scaledImage.cropped(to: cropRect)
    }
    
    private func runVisionInference(_ image: CIImage, model: VNCoreMLModel) -> MTLTexture? {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFit
        
        let handler = VNImageRequestHandler(ciImage: image)
        
        do {
            try handler.perform([request])
            
            guard let results = request.results,
                  let pixelBufferResult = results.first as? VNPixelBufferObservation else {
                print("❌ No results from Vision")
                return nil
            }
            
            return convertPixelBufferToTexture(pixelBufferResult.pixelBuffer)
            
        } catch {
            print("❌ Vision error: \(error)")
            return nil
        }
    }
    
    private func convertPixelBufferToTexture(_ pixelBuffer: CVPixelBuffer) -> MTLTexture? {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create texture descriptor
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,  // Single channel float for depth
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard let texture = metalDevice.makeTexture(descriptor: descriptor) else {
            print("❌ Failed to create texture")
            return nil
        }
        
        // Copy pixel buffer data to texture
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            print("❌ Failed to get pixel buffer base address")
            return nil
        }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let region = MTLRegionMake2D(0, 0, width, height)
        
        // Handle different pixel formats
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        
        if pixelFormat == kCVPixelFormatType_OneComponent32Float {
            // Direct float data
            texture.replace(region: region, mipmapLevel: 0, withBytes: baseAddress, bytesPerRow: bytesPerRow)
        } else {
            // Convert other formats to float
            print("⚠️ Unexpected pixel format: \(pixelFormat), attempting conversion")
            return convertToFloatTexture(pixelBuffer: pixelBuffer, targetTexture: texture)
        }
        
        return texture
    }
    
    private func convertToFloatTexture(pixelBuffer: CVPixelBuffer, targetTexture: MTLTexture) -> MTLTexture? {
        // Use CIContext to convert pixel buffer to float texture
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        let colorSpace = CGColorSpaceCreateDeviceGray()
        
        ciContext.render(
            ciImage,
            to: targetTexture,
            commandBuffer: nil,
            bounds: ciImage.extent,
            colorSpace: colorSpace
        )
        
        return targetTexture
    }
    
    private func normalizeDepthSimple(_ depthTexture: MTLTexture) -> MTLTexture? {
        // Create simple normalization compute shader
        guard let library = metalDevice.makeDefaultLibrary(),
              let function = library.makeFunction(name: "simpleNormalizeDepth"),
              let pipelineState = try? metalDevice.makeComputePipelineState(function: function) else {
            print("⚠️ Using raw depth without normalization")
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
}
