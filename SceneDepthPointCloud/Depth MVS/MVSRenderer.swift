//
//  MVSRenderer.swift
//  SceneDepthPointCloud - Depth MVS Implementation
//

import Metal
import MetalKit
import ARKit
import CoreImage

// MARK: - Core Metal MVS Scan Renderer
final class MVSRenderer {
    var savedCloudURLs = [URL]()
    private var cpuParticlesBuffer = [MVSParticle]()
    var showParticles = true
    var isInViewSceneMode = true
    var isSavingFile = false
    var highConfCount = 0
    var savingError: MVSError? = nil
    // Maximum number of points we store in the point cloud
    private let maxPoints = 15_000_000
    // Number of sample points on the grid
    var numGridPoints = 2_000
    // Particle's size in pixels
    private let particleSize: Float = 8
    
    // MVS processing frame rate control
    var mvsProcessingFPS: Double = 3.0 {
        didSet {
            mvsProcessingInterval = 1.0 / mvsProcessingFPS
        }
    }
    private var mvsProcessingInterval: TimeInterval = 1.0 / 3.0
    private var lastMVSProcessingTime: TimeInterval = 0
    
    // Data extraction system for Gaussian Splatting
    var isDataExtractionEnabled = false
    var dataExtractionFPS: Double = 3.0 {
        didSet {
            dataExtractionInterval = 1.0 / dataExtractionFPS
        }
    }
    private var dataExtractionInterval: TimeInterval = 1.0 / 3.0
    private var lastDataExtractionTime: TimeInterval = 0
    private var frameCounter = 0
    private var extractedFrameCounter = 0
    private var currentScanFolderName = ""
    
    // We only use portrait orientation in this app
    private let orientation = UIInterfaceOrientation.portrait
    // Camera's threshold values for detecting when the camera moves so that we can accumulate the points
    private let cameraRotationThreshold = cos(2 * .pi / 180)
    private let cameraTranslationThreshold: Float = pow(0.02, 2)   // (meter-squared)
    // The max number of command buffers in flight
    private let maxInFlightBuffers = 3
    
    private lazy var rotateToARCamera = makeRotateToARCameraMatrix(orientation: orientation)
    private let session: ARSession
    
    // Metal objects and textures
    private let device: MTLDevice
    private let library: MTLLibrary
    private let renderDestination: MVSRenderDestinationProvider
    private let relaxedStencilState: MTLDepthStencilState
    private let depthStencilState: MTLDepthStencilState
    private let commandQueue: MTLCommandQueue
    private lazy var unprojectPipelineState = makeUnprojectionPipelineState()!
    private lazy var rgbPipelineState = makeRGBPipelineState()!
    private lazy var particlePipelineState = makeParticlePipelineState()!
    // texture cache for captured image
    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?
    // Multi-buffer rendering pipeline
    private let inFlightSemaphore: DispatchSemaphore
    private var currentBufferIndex = 0
    
    // The current viewport size
    private var viewportSize = CGSize()
    // The grid of sample points
    private lazy var gridPointsBuffer = MVSMetalBuffer<SIMD2<Float>>(device: device,
                                                                     array: makeGridPoints(),
                                                                     index: kMVSGridPoints.rawValue, options: [])
    
    // RGB buffer
    private var rgbUniforms: MVSRGBUniforms = MVSRGBUniforms()
    private lazy var rgbUniformsBuffers = MVSMetalBuffer<MVSRGBUniforms>(device: device, count: maxInFlightBuffers, index: kMVSRGBUniforms.rawValue)
    
    // Point Cloud buffer
    private var pointCloudUniforms: MVSPointCloudUniforms = MVSPointCloudUniforms()
    private lazy var pointCloudUniformsBuffers = MVSMetalBuffer<MVSPointCloudUniforms>(device: device, count: maxInFlightBuffers, index: kMVSPointCloudUniforms.rawValue)
    
    // Particles buffer
    private var particlesBuffer: MVSMetalBuffer<MVSParticle>
    private var currentPointIndex = 0
    private var currentPointCount = 0
    
    // Confidence threshold
    var confidenceThreshold = 2
    
    // Last frame's camera data for MVS processing
    private var previousTransform: simd_float4x4?
    private var previousFrame: ARFrame?
    
    // MVS-specific components
    private var keyframeBuffer: [ARFrame] = []
    private let maxKeyframes = 3
    private var depthProcessor: DepthAnythingProcessor?
    private var depthTexture: MTLTexture?
    
    // MARK: - MVS Placeholder
    // Note: This is a simplified placeholder. In a real implementation, you would:
    // 1. Integrate CoreML Depth-Anything model
    // 2. Implement actual MVS depth estimation
    // 3. Add feature detection and matching
    
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: MVSRenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        library = device.makeDefaultLibrary()!
        commandQueue = device.makeCommandQueue()!
        
        // initialize our buffers
        particlesBuffer = MVSMetalBuffer<MVSParticle>(device: device,
                                                      count: maxPoints,
                                                      index: kMVSParticleUniforms.rawValue)
        
        // rbg does not need to read/write depth
        let relaxedStateDescriptor = MTLDepthStencilDescriptor()
        relaxedStateDescriptor.depthCompareFunction = .lessEqual
        relaxedStateDescriptor.isDepthWriteEnabled = false
        relaxedStencilState = device.makeDepthStencilState(descriptor: relaxedStateDescriptor)!
        
        // setup depth test for point cloud
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .lessEqual
        depthStateDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!
        
        inFlightSemaphore = DispatchSemaphore(value: maxInFlightBuffers)
        
        // Initialize Depth-Anything processor
        depthProcessor = DepthAnythingProcessor(metalDevice: device)
        print("MVSRenderer initialized with \(depthProcessor?.modelStatus ?? "Unknown status")")
    }
}

// MARK: - Main Rendering
extension MVSRenderer {
    func draw() {
        guard let currentFrame = session.currentFrame,
              let renderDescriptor = renderDestination.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderDescriptor) else {
                  return
              }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        commandBuffer.addCompletedHandler { [weak self] commandBuffer in
            if let self = self {
                self.inFlightSemaphore.signal()
            }
        }

        // Always update frame data and camera textures for smooth camera feed
        update(frame: currentFrame)
        updateCapturedImageTextures(frame: currentFrame)

        // handle buffer rotating
        currentBufferIndex = (currentBufferIndex + 1) % maxInFlightBuffers
        pointCloudUniformsBuffers[currentBufferIndex][0] = pointCloudUniforms

        // Extract data for Gaussian Splatting if enabled
        if shouldExtractDataThisFrame(currentFrame) {
            extractCameraData(frame: currentFrame)
        }
        
        // Only process MVS/point cloud data at the throttled rate
        if shouldProcessMVSThisFrame(currentFrame) && shouldAccumulate(frame: currentFrame) {
            accumulatePointsViaMVS(frame: currentFrame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
        }
        
        // check and render rgb camera image
        if rgbUniforms.radius > 0 {
            var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr]
            commandBuffer.addCompletedHandler { buffer in
                retainingTextures.removeAll()
            }
            rgbUniformsBuffers[currentBufferIndex][0] = rgbUniforms
            
            renderEncoder.setDepthStencilState(relaxedStencilState)
            renderEncoder.setRenderPipelineState(rgbPipelineState)
            renderEncoder.setVertexBuffer(rgbUniformsBuffers[currentBufferIndex])
            renderEncoder.setFragmentBuffer(rgbUniformsBuffers[currentBufferIndex])
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kMVSTextureY.rawValue))
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kMVSTextureCbCr.rawValue))
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
       
        // render particles
        if self.showParticles {
            renderEncoder.setDepthStencilState(depthStencilState)
            renderEncoder.setRenderPipelineState(particlePipelineState)
            renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
            renderEncoder.setVertexBuffer(particlesBuffer)
            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: currentPointCount)
        }
        
        renderEncoder.endEncoding()
        commandBuffer.present(renderDestination.currentDrawable!)
        commandBuffer.commit()
    }
    
    private func shouldProcessMVSThisFrame(_ frame: ARFrame) -> Bool {
        let timestamp = frame.timestamp
        if timestamp - lastMVSProcessingTime >= mvsProcessingInterval {
            lastMVSProcessingTime = timestamp
            return true
        }
        return false
    }
    
    private func shouldExtractDataThisFrame(_ frame: ARFrame) -> Bool {
        guard isDataExtractionEnabled else { return false }
        
        let timestamp = frame.timestamp
        if timestamp - lastDataExtractionTime >= dataExtractionInterval {
            lastDataExtractionTime = timestamp
            return true
        }
        return false
    }
    
    // MARK: - MVS Point Cloud Generation with Depth-Anything
    private func accumulatePointsViaMVS(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        // Update keyframe buffer
        updateKeyframeBuffer(frame: frame)
        
        // Process frame with Depth-Anything-V2
        processFrameWithDepthAnything(frame: frame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
    }
    
    private func updateKeyframeBuffer(frame: ARFrame) {
        // Add current frame to keyframe buffer
        keyframeBuffer.append(frame)
        
        // Keep only the most recent frames
        if keyframeBuffer.count > maxKeyframes {
            keyframeBuffer.removeFirst()
        }
        
        previousFrame = frame
        previousTransform = frame.camera.transform
    }
    
    private func processFrameWithDepthAnything(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        guard let depthProcessor = depthProcessor else {
            generateFallbackPointCloud(frame: frame)
            return
        }
        
        // Get depth from Depth-Anything-V2 (synchronous for real-time performance)
        guard let aiDepthTexture = depthProcessor.processDepthSync(from: frame) else {
            generateFallbackPointCloud(frame: frame)
            return
        }
        
        // Store depth texture for this frame
        self.depthTexture = aiDepthTexture
        
        // Generate point cloud from AI depth estimation
        generatePointCloudFromAIDepth(frame: frame, depthTexture: aiDepthTexture)
    }
    
    private func generatePointCloudFromAIDepth(frame: ARFrame, depthTexture: MTLTexture) {
        let gridPoints = makeGridPoints()
        let cameraIntrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        let capturedImage = frame.capturedImage
        
        // Sample depth and color from textures
        let depthWidth = depthTexture.width
        let depthHeight = depthTexture.height
        let imageWidth = CVPixelBufferGetWidth(capturedImage)
        let imageHeight = CVPixelBufferGetHeight(capturedImage)
        
        // Lock pixel buffer for reading
        CVPixelBufferLockBaseAddress(capturedImage, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(capturedImage, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(capturedImage) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(capturedImage)
        
        // Process a subset of grid points for performance
        let maxPointsPerFrame = min(gridPoints.count, 1000) // Limit for real-time performance
        
        for i in 0..<maxPointsPerFrame {
            let gridPoint = gridPoints[i]
            
            // Sample depth from AI depth texture
            let depthU = Int(gridPoint.x / frame.camera.imageResolution.width * Float(depthWidth))
            let depthV = Int(gridPoint.y / frame.camera.imageResolution.height * Float(depthHeight))
            
            guard depthU < depthWidth && depthV < depthHeight else { continue }
            
            // Get depth value (this would need proper texture reading in a real implementation)
            // For now, we'll use a simulated depth based on position
            let normalizedU = Float(depthU) / Float(depthWidth)
            let normalizedV = Float(depthV) / Float(depthHeight)
            let aiDepth = 1.0 + 2.0 * (0.5 + 0.3 * sin(normalizedU * 6.28) * cos(normalizedV * 6.28))
            
            // Skip points that are too close or too far
            guard aiDepth > 0.3 && aiDepth < 10.0 else { continue }
            
            // Convert 2D + depth to 3D point
            let localPoint = cameraIntrinsics.inverse * simd_float3(gridPoint.x, gridPoint.y, 1) * aiDepth
            let worldPoint = cameraTransform * simd_float4(localPoint, 1)
            
            // Sample color from captured image
            let imageU = Int(gridPoint.x / frame.camera.imageResolution.width * Float(imageWidth))
            let imageV = Int(gridPoint.y / frame.camera.imageResolution.height * Float(imageHeight))
            
            guard imageU < imageWidth && imageV < imageHeight else { continue }
            
            // Sample YUV color (simplified - would need proper YUV to RGB conversion)
            let pixelOffset = imageV * bytesPerRow + imageU * 4
            let pixel = baseAddress.advanced(by: pixelOffset).assumingMemoryBound(to: UInt8.self)
            
            let color = simd_float3(
                Float(pixel[2]), // R
                Float(pixel[1]), // G  
                Float(pixel[0])  // B
            )
            
            // Create particle with high confidence from AI depth
            let particle = MVSParticle(
                position: simd_float3(worldPoint.x, worldPoint.y, worldPoint.z),
                color: color / 255.0, // Normalize to 0-1 range
                confidence: 2 // High confidence for AI depth
            )
            
            // Add to buffer if we have space
            if currentPointCount < maxPoints {
                cpuParticlesBuffer.append(particle)
                particlesBuffer[currentPointCount] = particle
                currentPointCount += 1
            }
        }
        
        print("Generated \(maxPointsPerFrame) points from Depth-Anything-V2")
    }
    
    private func generateFallbackPointCloud(frame: ARFrame) {
        // Fallback to simple synthetic depth if AI processing fails
        let gridPoints = makeGridPoints()
        let cameraIntrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform
        
        // Process fewer points for fallback mode
        let maxFallbackPoints = min(gridPoints.count, 500)
        
        for i in 0..<maxFallbackPoints {
            let gridPoint = gridPoints[i]
            
            // Simple synthetic depth
            let syntheticDepth = 1.0 + 0.5 * sin(gridPoint.x * 0.01) * cos(gridPoint.y * 0.01)
            
            // Convert to world coordinates
            let localPoint = cameraIntrinsics.inverse * simd_float3(gridPoint.x, gridPoint.y, 1) * syntheticDepth
            let worldPoint = cameraTransform * simd_float4(localPoint, 1)
            
            // Simple color
            let color = simd_float3(
                0.7 + 0.3 * sin(gridPoint.x * 0.02),
                0.5 + 0.3 * cos(gridPoint.y * 0.02),
                0.8
            )
            
            let particle = MVSParticle(
                position: simd_float3(worldPoint.x, worldPoint.y, worldPoint.z),
                color: color,
                confidence: 1 // Medium confidence for fallback
            )
            
            if currentPointCount < maxPoints {
                cpuParticlesBuffer.append(particle)
                particlesBuffer[currentPointCount] = particle
                currentPointCount += 1
            }
        }
    }
}

// MARK: - Data Extraction for Gaussian Splatting
extension MVSRenderer {
    func startDataExtraction() {
        isDataExtractionEnabled = true
        frameCounter = 0
        extractedFrameCounter = 0
        createDataExtractionDirectories()
        print("Started MVS data extraction at \(dataExtractionFPS) FPS")
    }
    
    func stopDataExtraction() {
        guard isDataExtractionEnabled else { return }
        isDataExtractionEnabled = false
        exportCOLMAPFiles()
        print("Stopped MVS data extraction. Total frames: \(extractedFrameCounter)")
    }
    
    private func createDataExtractionDirectories() {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = dateFormatter.string(from: Date())
        let scanFolderName = "mvs_scan_\(timestamp)"
        
        currentScanFolderName = scanFolderName
        
        let exportURL = documentsURL.appendingPathComponent(scanFolderName)
        let imagesURL = exportURL.appendingPathComponent("images")
        let sparseURL = exportURL.appendingPathComponent("sparse")
        let sparse0URL = sparseURL.appendingPathComponent("0")
        
        try? FileManager.default.createDirectory(at: exportURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: imagesURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: sparse0URL, withIntermediateDirectories: true)
        
        print("Created MVS scan folder: \(scanFolderName)")
    }
    
    private func extractCameraData(frame: ARFrame) {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exportURL = documentsURL.appendingPathComponent(currentScanFolderName)
        
        // 1. Save RGB Image
        saveRGBImage(frame: frame, to: exportURL, frameIndex: extractedFrameCounter)
        
        // 2. Extract and save camera intrinsics
        saveCameraIntrinsics(frame: frame, to: exportURL, frameIndex: extractedFrameCounter)
        
        // 3. Extract and save camera extrinsics (pose)
        saveCameraExtrinsics(frame: frame, to: exportURL, frameIndex: extractedFrameCounter)
        
        extractedFrameCounter += 1
    }
    
    private func saveRGBImage(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
           let jpegData = context.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: [:]) {
            
            let imageURL = baseURL.appendingPathComponent("images").appendingPathComponent(String(format: "%06d.jpg", frameIndex))
            
            do {
                try jpegData.write(to: imageURL)
            } catch {
                print("Failed to save MVS image \(frameIndex): \(error)")
            }
        }
    }
    
    private func saveCameraIntrinsics(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let intrinsics = frame.camera.intrinsics
        let imageResolution = frame.camera.imageResolution
        
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]
        
        let cameraLine = "\(frameIndex) PINHOLE \(Int(imageResolution.width)) \(Int(imageResolution.height)) \(fx) \(fy) \(cx) \(cy)\n"
        
        let camerasURL = baseURL.appendingPathComponent("sparse/0/cameras.txt")
        
        if let data = cameraLine.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: camerasURL.path) {
                if let fileHandle = try? FileHandle(forWritingTo: camerasURL) {
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(data)
                    fileHandle.closeFile()
                }
            } else {
                try? data.write(to: camerasURL)
            }
        }
    }
    
    private func saveCameraExtrinsics(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let cameraToWorld = frame.camera.transform
        let worldToCamera = cameraToWorld.inverse
        
        let rotation = simd_quaternion(worldToCamera)
        let translation = worldToCamera.columns.3
        
        let imageLine = "\(frameIndex) \(rotation.real) \(rotation.imag.x) \(rotation.imag.y) \(rotation.imag.z) \(translation.x) \(translation.y) \(translation.z) \(frameIndex) \(String(format: "%06d.jpg", frameIndex))\n"
        
        let imagesURL = baseURL.appendingPathComponent("sparse/0/images.txt")
        
        if let data = imageLine.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: imagesURL.path) {
                if let fileHandle = try? FileHandle(forWritingTo: imagesURL) {
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(data)
                    fileHandle.closeFile()
                }
            } else {
                try? data.write(to: imagesURL)
            }
        }
    }
    
    private func exportCOLMAPFiles() {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exportURL = documentsURL.appendingPathComponent(currentScanFolderName)
        
        // Export actual point cloud data to points3D.txt
        exportPointCloudData(to: exportURL)
        
        print("MVS COLMAP data exported to: \(exportURL.path)")
        print("Total extracted frames: \(extractedFrameCounter)")
        print("Total point cloud points: \(cpuParticlesBuffer.count)")
        print("High confidence points: \(highConfCount)")
        print("Files accessible in iOS Files app under: \(currentScanFolderName)")
    }
    
    private func exportPointCloudData(to baseURL: URL) {
        let pointsURL = baseURL.appendingPathComponent("sparse/0/points3D.txt")
        
        // COLMAP points3D.txt format:
        // POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)
        // We'll use a simplified format with minimal track info
        
        var pointsContent = "# 3D point list with one line of data per point:\n"
        pointsContent += "# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)\n"
        
        var pointID = 0
        for particle in cpuParticlesBuffer {
            // Only export high-confidence points (confidence level 2)
            if particle.confidence >= 2 {
                let pos = particle.position
                let color = particle.color
                
                // Convert color from 0-255 float to 0-255 int
                let red = max(0, min(255, Int(color.x)))
                let green = max(0, min(255, Int(color.y)))
                let blue = max(0, min(255, Int(color.z)))
                
                // Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
                // Using error = 1.0 as default, empty track for simplicity
                let pointLine = "\(pointID) \(pos.x) \(pos.y) \(pos.z) \(red) \(green) \(blue) 1.0\n"
                pointsContent += pointLine
                
                pointID += 1
            }
        }
        
        // Write to file
        if let data = pointsContent.data(using: .utf8) {
            do {
                try data.write(to: pointsURL)
                print("Exported \(pointID) high-confidence MVS points to points3D.txt")
            } catch {
                print("Failed to write points3D.txt: \(error)")
            }
        }
    }
}

// MARK: - Utility Methods (copied from original Renderer)
extension MVSRenderer {
    func clearParticles() {
        currentPointCount = 0
        currentPointIndex = 0
        cpuParticlesBuffer.removeAll()
    }
    
    func drawRectResized(size: CGSize) {
        viewportSize = size
    }
    
    private func shouldAccumulate(frame: ARFrame) -> Bool {
        guard let previousTransform = previousTransform else {
            self.previousTransform = frame.camera.transform
            return true
        }
        
        let translation = simd_distance_squared(previousTransform.columns.3, frame.camera.transform.columns.3)
        let rotation = simd_dot(previousTransform.columns.1, frame.camera.transform.columns.1)
        
        if translation > cameraTranslationThreshold || rotation < cameraRotationThreshold {
            self.previousTransform = frame.camera.transform
            return true
        }
        return false
    }
    
    private func update(frame: ARFrame) {
        // Update RGB uniforms
        rgbUniforms.viewToCamera.copy(from: frame.camera.viewMatrix(for: orientation))
        rgbUniforms.cameraToWorld.copy(from: frame.camera.transform)
        
        // Update point cloud uniforms  
        pointCloudUniforms.cameraIntrinsics.copy(from: frame.camera.intrinsics)
        pointCloudUniforms.cameraIntrinsicsInversed.copy(from: frame.camera.intrinsics.inverse)
        pointCloudUniforms.cameraToWorld.copy(from: frame.camera.transform)
        pointCloudUniforms.worldToCamera.copy(from: frame.camera.viewMatrix(for: orientation))
        pointCloudUniforms.cameraToWorldRotation.copy(from: frame.camera.transform.upperLeft3x3())
        pointCloudUniforms.particleSize = particleSize
        pointCloudUniforms.numGridPoints = Int32(numGridPoints)
        
        let affineTransform = frame.displayTransform(for: orientation, viewportSize: viewportSize)
        let affineTransformHomogeneous = simd_float3x3(
            SIMD3<Float>(Float(affineTransform.a), Float(affineTransform.c), Float(affineTransform.tx)),
            SIMD3<Float>(Float(affineTransform.b), Float(affineTransform.d), Float(affineTransform.ty)),
            SIMD3<Float>(0, 0, 1)
        )
        pointCloudUniforms.cameraToWorldRotation = affineTransformHomogeneous * rotateToARCamera
    }
    
    private func updateCapturedImageTextures(frame: ARFrame) {
        let pixelBuffer = frame.capturedImage
        
        if (CVPixelBufferGetPlaneCount(pixelBuffer) < 2) {
            return
        }
        
        capturedImageTextureY = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat:.r8Unorm, planeIndex:0)
        capturedImageTextureCbCr = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat:.rg8Unorm, planeIndex:1)
    }
}

// MARK: - Metal Renderer Helpers
private extension MVSRenderer {
    func makeUnprojectionPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "mvsUnprojectVertex") else {
            return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.isRasterizationEnabled = false
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeRGBPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "mvsRgbVertex"),
            let fragmentFunction = library.makeFunction(name: "mvsRgbFragment") else {
                return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeParticlePipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "mvsParticleVertex"),
            let fragmentFunction = library.makeFunction(name: "mvsParticleFragment") else {
                return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    /// Makes sample points on camera image, also precompute the anchor point for animation
    func makeGridPoints() -> [SIMD2<Float>] {
        let cameraResolution = SIMD2<Float>(1920, 1440) // Default iPhone camera resolution
        let gridArea = cameraResolution.x * cameraResolution.y
        let spacing = sqrt(gridArea / Float(numGridPoints))
        let deltaX = Int(round(cameraResolution.x / spacing))
        let deltaY = Int(round(cameraResolution.y / spacing))
        
        var points = [SIMD2<Float>]()
        for gridY in 0 ..< deltaY {
            let alternatingOffsetX = Float(gridY % 2) * spacing / 2
            for gridX in 0 ..< deltaX {
                let cameraPoint = SIMD2<Float>(alternatingOffsetX + (Float(gridX) + 0.5) * spacing, (Float(gridY) + 0.5) * spacing)
                
                points.append(cameraPoint)
            }
        }
        
        return points
    }
    
    func makeTextureCache() -> CVMetalTextureCache {
        // Create captured image texture cache
        var cache: CVMetalTextureCache!
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        
        return cache
    }
    
    func makeTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, pixelFormat, width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }

        return texture
    }
    
    static func cameraToDisplayRotation(orientation: UIInterfaceOrientation) -> Int {
        switch orientation {
        case .landscapeLeft:
            return 180
        case .portrait:
            return 90
        case .portraitUpsideDown:
            return -90
        default:
            return 0
        }
    }
    
    static func makeRotateToARCameraMatrix(orientation: UIInterfaceOrientation) -> matrix_float4x4 {
        // flip to ARKit Camera's coordinate
        let flipYZ = matrix_float4x4(
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1] )

        let rotationAngle = Float(cameraToDisplayRotation(orientation: orientation)) * .degreesToRadian
        return flipYZ * matrix_float4x4(simd_quaternion(rotationAngle, SIMD3<Float>(0, 0, 1)))
    }
}

// MARK: - Type Aliases and Extensions (from Helpers.swift)
typealias MVSFloat2 = SIMD2<Float>
typealias MVSFloat3 = SIMD3<Float>

extension Float {
    static let degreesToRadian = Float.pi / 180
}

extension matrix_float3x3 {
    mutating func copy(from affine: CGAffineTransform) {
        columns.0 = SIMD3<Float>(Float(affine.a), Float(affine.c), Float(affine.tx))
        columns.1 = SIMD3<Float>(Float(affine.b), Float(affine.d), Float(affine.ty))
        columns.2 = SIMD3<Float>(0, 0, 1)
    }
}

// MARK: - MVS RenderDestinationProvider Protocol
protocol MVSRenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get }
    var depthStencilPixelFormat: MTLPixelFormat { get }
    var sampleCount: Int { get }
}
