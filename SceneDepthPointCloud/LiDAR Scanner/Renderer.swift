//
//  Renderer.swift
//  SceneDepthPointCloud

import Metal
import MetalKit
import ARKit
import CoreImage
import Foundation
import Photos


// MARK: - Core Metal Scan Renderer
final class Renderer {
    var savedCloudURLs = [URL]()
    private var cpuParticlesBuffer = [CPUParticle]()
    var showParticles = true
    var isInViewSceneMode = true
    var isSavingFile = false
    var highConfCount = 0
    var savingError: XError? = nil
    // Maximum number of points we store in the point cloud
    private let maxPoints = 15_000_000
    // Number of sample points on the grid
    var numGridPoints = 2_200  // Increased by 10% from 2000
    // Particle's size in pixels
    private let particleSize: Float = 8

    // LiDAR/Point cloud processing frame rate control
    var lidarProcessingFPS: Double = 3.0 {
        didSet {
            lidarProcessingInterval = 1.0 / lidarProcessingFPS
        }
    }
    private var lidarProcessingInterval: TimeInterval = 1.0 / 3.0
    private var lastLidarProcessingTime: TimeInterval = 0

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
    // set to 0 for continous sampling
    private let cameraRotationThreshold = cos(0 * .degreesToRadian)
    private let cameraTranslationThreshold: Float = pow(0.00, 2)   // (meter-squared)
    // The max number of command buffers in flight
    private let maxInFlightBuffers = 5

    private lazy var rotateToARCamera = Self.makeRotateToARCameraMatrix(orientation: orientation)
    private let session: ARSession

    // Metal objects and textures
    private let device: MTLDevice
    private let library: MTLLibrary
    private let renderDestination: RenderDestinationProvider
    private let relaxedStencilState: MTLDepthStencilState
    private let depthStencilState: MTLDepthStencilState
    private var commandQueue: MTLCommandQueue
    private lazy var unprojectPipelineState = makeUnprojectionPipelineState()!
    private lazy var rgbPipelineState = makeRGBPipelineState()!
    private lazy var particlePipelineState = makeParticlePipelineState()!
    // texture cache for captured image
    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?
    private var depthTexture: CVMetalTexture?
    private var confidenceTexture: CVMetalTexture?

    // Multi-buffer rendering pipeline
    private let inFlightSemaphore: DispatchSemaphore
    private var currentBufferIndex = 0

    // The current viewport size
    private var viewportSize = CGSize()
    // The grid of sample points
    private lazy var gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                                            array: makeGridPoints(),
                                                            index: kGridPoints.rawValue, options: [])

    // RGB buffer
    private lazy var rgbUniforms: RGBUniforms = {
        var uniforms = RGBUniforms()
        uniforms.radius = rgbOn ? 2 : 0
        uniforms.viewToCamera.copy(from: viewToCamera)
        uniforms.viewRatio = Float(viewportSize.width / viewportSize.height)
        return uniforms
    }()
    private var rgbUniformsBuffers = [MetalBuffer<RGBUniforms>]()
    // Point Cloud buffer
    private lazy var pointCloudUniforms: PointCloudUniforms = {
        var uniforms = PointCloudUniforms()
        uniforms.maxPoints = Int32(maxPoints)
        uniforms.confidenceThreshold = Int32(confidenceThreshold)
        uniforms.particleSize = particleSize
        uniforms.cameraResolution = cameraResolution
        return uniforms
    }()
    private var pointCloudUniformsBuffers = [MetalBuffer<PointCloudUniforms>]()
    // Particles buffer
    private var particlesBuffer: MetalBuffer<ParticleUniforms>
    private var currentPointIndex = 0
    private var currentPointCount = 0

    // Camera data
    private var sampleFrame: ARFrame { session.currentFrame! }
    private lazy var cameraResolution = Float2(Float(sampleFrame.camera.imageResolution.width), Float(sampleFrame.camera.imageResolution.height))
    private lazy var viewToCamera = sampleFrame.displayTransform(for: orientation, viewportSize: viewportSize).inverted()
    private lazy var lastCameraTransform = sampleFrame.camera.transform

    // interfaces
    var confidenceThreshold = 2
    
    // Depth source selection
    var depthSource: DepthSource = .lidar
    
    // PromptDA + LoG sampling support
    private var promptDAEngine: PromptDAEngine?
    private var cachedProbSamples: [Float2]?
    
    // Depth photo saving
    private var depthPhotoCounter = 0
    private var shouldSaveNextDepth = false

    var rgbOn: Bool = false {
        didSet {
            // apply the change for the shader
            rgbUniforms.radius = rgbOn ? 2 : 0
        }
    }

    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        library = device.makeDefaultLibrary()!

        commandQueue = device.makeCommandQueue()!
        // initialize our buffers
        for _ in 0 ..< maxInFlightBuffers {
            rgbUniformsBuffers.append(.init(device: device, count: 1, index: 0))
            pointCloudUniformsBuffers.append(.init(device: device, count: 1, index: kPointCloudUniforms.rawValue))
        }
        particlesBuffer = .init(device: device, count: maxPoints, index: kParticleUniforms.rawValue)
        // rbg does not need to read/write depth
        let relaxedStateDescriptor = MTLDepthStencilDescriptor()
        relaxedStencilState = device.makeDepthStencilState(descriptor: relaxedStateDescriptor)!

        // setup depth test for point cloud
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .lessEqual
        depthStateDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!

        inFlightSemaphore = DispatchSemaphore(value: maxInFlightBuffers)
        
        // Try to initialize PromptDA engine (will be used if depthSource == .mvs)
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🔧 Renderer: Initializing PromptDA Engine...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        do {
            let startTime = CFAbsoluteTimeGetCurrent()
            // Use 256×192 prompt size (matches ARKit depth directly, no rotation)
            promptDAEngine = try PromptDAEngine.create(
                bundleModelName: "PromptDA_vits_518x518_prompt256x192",
                rgbSize: .init(width: 518, height: 518),
                promptHW: (256, 192)
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("✅ Renderer: PromptDA ready! (loaded in \(String(format: "%.2f", elapsed))s)")
            print("   MVS mode is available")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        } catch {
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("⚠️ Renderer: PromptDA NOT available")
            print("   Error: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("   Domain: \(nsError.domain)")
                print("   Code: \(nsError.code)")
                if let reason = nsError.userInfo[NSLocalizedFailureReasonErrorKey] as? String {
                    print("   Reason: \(reason)")
                }
            }
            print("   → MVS mode will fall back to LiDAR")
            print("   → Add the PromptDA .mlpackage to Xcode project to enable MVS")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        }
        
        self.loadSavedClouds()
    }

    func drawRectResized(size: CGSize) {
        viewportSize = size
    }

    private func updateCapturedImageTextures(frame: ARFrame) {
        // Create two textures (Y and CbCr) from the provided frame's captured image
        let pixelBuffer = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 else {
            return
        }

        capturedImageTextureY = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0)
        capturedImageTextureCbCr = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1)
    }

    private func updateDepthTextures(frame: ARFrame) -> Bool {
        // Use different depth processing based on depth source
        switch depthSource {
        case .mvs:
            return updateMVSDepthTextures(frame: frame)
        case .lidar, .depthView:
            return updateLiDARDepthTextures(frame: frame)
        }
    }
    
    private func updateLiDARDepthTextures(frame: ARFrame) -> Bool {
        guard let depthMap = frame.smoothedSceneDepth?.depthMap,
            let confidenceMap = frame.smoothedSceneDepth?.confidenceMap else {
                print("⚠️ LiDAR: No smoothed scene depth available")
                return false
        }

        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)
        let confW = CVPixelBufferGetWidth(confidenceMap)
        let confH = CVPixelBufferGetHeight(confidenceMap)
        
        print("📍 LiDAR Mode:")
        print("   • Smoothed depth resolution: \(depthW)×\(depthH)")
        print("   • Confidence map resolution: \(confW)×\(confH)")
        print("   • Camera resolution: \(Int(cameraResolution.x))×\(Int(cameraResolution.y))")
        print("   • Grid points: \(numGridPoints) (uniform grid)")

        depthTexture = makeTexture(fromPixelBuffer: depthMap, pixelFormat: .r32Float, planeIndex: 0)
        confidenceTexture = makeTexture(fromPixelBuffer: confidenceMap, pixelFormat: .r8Uint, planeIndex: 0)
        
        // Save depth photo if requested
        if shouldSaveNextDepth {
            saveDepthAsPhoto(depthMap)
            shouldSaveNextDepth = false
        }
        
        // Clear cached probability samples (use uniform grid for LiDAR)
        cachedProbSamples = nil

        return true
    }
    
    // Save ARKit depth map as a photo in Photos app
    private func saveDepthAsPhoto(_ depthPixelBuffer: CVPixelBuffer) {
        depthPhotoCounter += 1
        
        print("📸 Saving ARKit depth map as photo...")
        
        // Convert depth to visible image (normalize to 0-255)
        let width = CVPixelBufferGetWidth(depthPixelBuffer)
        let height = CVPixelBufferGetHeight(depthPixelBuffer)
        
        CVPixelBufferLockBaseAddress(depthPixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthPixelBuffer, .readOnly) }
        
        let stride = CVPixelBufferGetBytesPerRow(depthPixelBuffer) / MemoryLayout<Float>.stride
        let depthData = CVPixelBufferGetBaseAddress(depthPixelBuffer)!.assumingMemoryBound(to: Float.self)
        
        // Find min/max for normalization
        var minDepth: Float = .infinity
        var maxDepth: Float = -.infinity
        for y in 0..<height {
            for x in 0..<width {
                let depth = depthData[y * stride + x]
                if depth > 0 && depth.isFinite {
                    minDepth = min(minDepth, depth)
                    maxDepth = max(maxDepth, depth)
                }
            }
        }
        
        print("   • Depth range: \(String(format: "%.3f", minDepth))m - \(String(format: "%.3f", maxDepth))m")
        
        // Create grayscale image
        var grayData = [UInt8](repeating: 0, count: width * height)
        let range = max(maxDepth - minDepth, 0.001)
        
        for y in 0..<height {
            for x in 0..<width {
                let depth = depthData[y * stride + x]
                let normalized = depth > 0 && depth.isFinite ? (depth - minDepth) / range : 0
                grayData[y * width + x] = UInt8(max(0, min(255, normalized * 255)))
            }
        }
        
        // Create CGImage from gray data
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let providerRef = CGDataProvider(data: Data(grayData) as CFData),
              let cgImage = CGImage(width: width,
                                   height: height,
                                   bitsPerComponent: 8,
                                   bitsPerPixel: 8,
                                   bytesPerRow: width,
                                   space: colorSpace,
                                   bitmapInfo: bitmapInfo,
                                   provider: providerRef,
                                   decode: nil,
                                   shouldInterpolate: false,
                                   intent: .defaultIntent) else {
            print("   ❌ Failed to create CGImage from depth data")
            return
        }
        
        let uiImage = UIImage(cgImage: cgImage)
        
        // Save to Photos
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized else {
                print("   ❌ Photos permission not granted")
                return
            }
            
            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: uiImage)
            }) { success, error in
                if success {
                    print("   ✅ Depth photo #\(self.depthPhotoCounter) saved to Photos (\(width)×\(height))")
                } else {
                    print("   ❌ Failed to save depth photo: \(error?.localizedDescription ?? "unknown error")")
                }
            }
        }
    }
    
    private func updateMVSDepthTextures(frame: ARFrame) -> Bool {
        guard let promptDA = promptDAEngine else {
            print("⚠️ PromptDA not available, falling back to LiDAR")
            return updateLiDARDepthTextures(frame: frame)
        }
        
        // Clear old particles on first MVS frame to avoid mixing LiDAR and MVS points
        // Check if we're transitioning from another mode
        if currentPointCount > 0 && cachedProbSamples == nil {
            print("🧹 Clearing old particles before first MVS frame (had \(currentPointCount) old points)")
            clearParticles()
        }
        
        // Log input data
        let rgbW = CVPixelBufferGetWidth(frame.capturedImage)
        let rgbH = CVPixelBufferGetHeight(frame.capturedImage)
        print("\n📍 MVS Mode (PromptDA):")
        print("   • Input RGB resolution: \(rgbW)×\(rgbH)")
        
        if let lidarPrompt = frame.smoothedSceneDepth?.depthMap {
            let promptW = CVPixelBufferGetWidth(lidarPrompt)
            let promptH = CVPixelBufferGetHeight(lidarPrompt)
            print("   • Input LiDAR prompt resolution: \(promptW)×\(promptH)")
        } else {
            print("   • No LiDAR prompt available")
        }
        
        do {
            // Run PromptDA with LiDAR as prompt (if available)
            let lidarPrompt = frame.smoothedSceneDepth?.depthMap
            let output = try promptDA.predict(rgbPB: frame.capturedImage, lidarPB: lidarPrompt)
            
            // Use PromptDA depth instead of LiDAR
            guard let newDepthTexture = makeTexture(fromPixelBuffer: output.depthPB, pixelFormat: .r32Float, planeIndex: 0) else {
                print("❌ Failed to create Metal texture from PromptDA depth output")
                return updateLiDARDepthTextures(frame: frame)
            }
            depthTexture = newDepthTexture
            
            // Generate LoG-based probability samples
            let depthW = CVPixelBufferGetWidth(output.depthPB)
            let depthH = CVPixelBufferGetHeight(output.depthPB)
            let probW = CVPixelBufferGetWidth(output.probPB)
            let probH = CVPixelBufferGetHeight(output.probPB)
            
            print("   • PromptDA output depth: \(depthW)×\(depthH)")
            print("   • LoG probability map: \(probW)×\(probH)")
            
            // Use Bernoulli sampling with scale factor of 2.0
            // Each pixel with LoG value p (0-1) is sampled with probability min(p * 2, 1.0)
            // This means edges (p≈1.0) → 100% sampled, flat regions (p≈0.1) → 20% sampled
            let scaleFactor: Float = 2.0
            let pixels = ProbabilitySampler.samplePixelsBernoulli(probPB: output.probPB, scale: scaleFactor)
            
            print("   • Sampled \(pixels.count) points from LoG (Bernoulli, scale=\(scaleFactor))")
            
            // Convert probability samples from PromptDA resolution to camera resolution
            // PromptDA outputs at 518×518, but camera is 1920×1440
            let scaleX = Float(cameraResolution.x) / Float(probW)
            let scaleY = Float(cameraResolution.y) / Float(probH)
            print("   • Scaling samples to camera resolution: scaleX=\(String(format: "%.2f", scaleX)), scaleY=\(String(format: "%.2f", scaleY))")
            
            cachedProbSamples = pixels.map { pixel in
                Float2(Float(pixel.x) * scaleX, Float(pixel.y) * scaleY)
            }
            
            print("   • Final sample count for rendering: \(cachedProbSamples?.count ?? 0)")
            
            // Use high confidence for PromptDA points (or reuse LiDAR confidence if available)
            if let confMap = frame.smoothedSceneDepth?.confidenceMap {
                confidenceTexture = makeTexture(fromPixelBuffer: confMap, pixelFormat: .r8Uint, planeIndex: 0)
                print("   • Using LiDAR confidence map")
            } else {
                // Generate synthetic high confidence map
                confidenceTexture = makeSyntheticConfidenceTexture(width: depthW, height: depthH)
                print("   • Using synthetic confidence map (all high confidence)")
            }
            
            // Ensure confidence texture was created
            if confidenceTexture == nil {
                print("❌ Failed to create confidence texture")
                return updateLiDARDepthTextures(frame: frame)
            }
            
            print("✅ PromptDA MVS complete\n")
            return true
            
        } catch {
            print("❌ PromptDA failed: \(error.localizedDescription), falling back to LiDAR")
            cachedProbSamples = nil
            return updateLiDARDepthTextures(frame: frame)
        }
    }
    
    private func makeSyntheticConfidenceTexture(width: Int, height: Int) -> CVMetalTexture? {
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, 
                                        kCVPixelFormatType_OneComponent8, nil, &pb)
        guard status == kCVReturnSuccess, let pixelBuffer = pb else { return nil }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        
        let buf = CVPixelBufferGetBaseAddress(pixelBuffer)!.assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
        
        // Fill with high confidence (2)
        for y in 0..<height {
            for x in 0..<width {
                buf[y * stride + x] = 2
            }
        }
        
        return makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Uint, planeIndex: 0)
    }

    private func update(frame: ARFrame) {
        // frame dependent info
        let camera = frame.camera
        let cameraIntrinsicsInversed = camera.intrinsics.inverse
        let viewMatrix = camera.viewMatrix(for: orientation)
        let viewMatrixInversed = viewMatrix.inverse
        let projectionMatrix = camera.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: 0.001, zFar: 0)
        pointCloudUniforms.viewProjectionMatrix = projectionMatrix * viewMatrix
        pointCloudUniforms.localToWorld = viewMatrixInversed * rotateToARCamera
        pointCloudUniforms.cameraIntrinsicsInversed = cameraIntrinsicsInversed
    }

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

        // Only process LiDAR/point cloud data at the throttled rate
        if shouldProcessLiDARThisFrame(currentFrame) && shouldAccumulate(frame: currentFrame), updateDepthTextures(frame: currentFrame) {
            accumulatePoints(frame: currentFrame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
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
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }

        // render particles
        if self.showParticles {
            print("   🎨 Rendering \(currentPointCount) particles")
            renderEncoder.setDepthStencilState(depthStencilState)
            renderEncoder.setRenderPipelineState(particlePipelineState)
            renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
            renderEncoder.setVertexBuffer(particlesBuffer)
            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: currentPointCount)
        } else {
            print("   ⚠️ Particles rendering disabled (showParticles=false)")
        }

        renderEncoder.endEncoding()
        commandBuffer.present(renderDestination.currentDrawable!)
        commandBuffer.commit()
    }

    private func shouldProcessLiDARThisFrame(_ frame: ARFrame) -> Bool {
        let currentTime = frame.timestamp
        if currentTime - lastLidarProcessingTime >= lidarProcessingInterval {
            lastLidarProcessingTime = currentTime
            return true
        }
        return false
    }

    // Helper function to get current processing rate for UI feedback
    func getCurrentProcessingRate() -> String {
        return String(format: "%.1f FPS", lidarProcessingFPS)
    }

    private func shouldExtractDataThisFrame(_ frame: ARFrame) -> Bool {
        guard isDataExtractionEnabled else { return false }

        let currentTime = frame.timestamp
        if currentTime - lastDataExtractionTime >= dataExtractionInterval {
            lastDataExtractionTime = currentTime
            return true
        }
        return false
    }

    private func shouldAccumulate(frame: ARFrame) -> Bool {
        if self.isInViewSceneMode || !self.showParticles {
            return false
        }
        let cameraTransform = frame.camera.transform
        return currentPointCount == 0
            || dot(cameraTransform.columns.2, lastCameraTransform.columns.2) <= cameraRotationThreshold
            || distance_squared(cameraTransform.columns.3, lastCameraTransform.columns.3) >= cameraTranslationThreshold
    }

    private func accumulatePoints(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        pointCloudUniforms.pointCloudCurrentIndex = Int32(currentPointIndex)

        var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr, depthTexture, confidenceTexture]

        commandBuffer.addCompletedHandler { buffer in
            retainingTextures.removeAll()
            // copy gpu point buffer to cpu
            var i = self.cpuParticlesBuffer.count
            while (i < self.maxPoints && self.particlesBuffer[i].position != simd_float3(0.0,0.0,0.0)) {
                let position = self.particlesBuffer[i].position
                let color = self.particlesBuffer[i].color
                let confidence = self.particlesBuffer[i].confidence
                if confidence == 2 { self.highConfCount += 1 }
                self.cpuParticlesBuffer.append(
                    CPUParticle(position: position,
                                color: color,
                                confidence: confidence))
                i += 1
            }
        }
        
        // Regenerate grid points buffer if we have new probability samples (MVS mode)
        if depthSource == .mvs && cachedProbSamples != nil {
            let newPoints = makeGridPoints()
            print("   • Regenerating grid points buffer: \(newPoints.count) LoG-sampled points")
            gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                                  array: newPoints,
                                                  index: kGridPoints.rawValue, options: [])
        }

        renderEncoder.setDepthStencilState(relaxedStencilState)
        renderEncoder.setRenderPipelineState(unprojectPipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        renderEncoder.setVertexBuffer(gridPointsBuffer)
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(depthTexture!), index: Int(kTextureDepth.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(confidenceTexture!), index: Int(kTextureConfidence.rawValue))
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: gridPointsBuffer.count)
        
        print("   • Drawing \(gridPointsBuffer.count) grid points to accumulate")

        currentPointIndex = (currentPointIndex + gridPointsBuffer.count) % maxPoints
        currentPointCount = min(currentPointCount + gridPointsBuffer.count, maxPoints)
        print("   • Current accumulated points: \(currentPointCount)/\(maxPoints)")
        lastCameraTransform = frame.camera.transform
    }
}

// MARK:  - Added Renderer functionality
extension Renderer {
    func toggleParticles() {
        self.showParticles = !self.showParticles
    }
    func toggleSceneMode() {
        self.isInViewSceneMode = !self.isInViewSceneMode
    }
    func getCpuParticles() -> Array<CPUParticle> {
        return self.cpuParticlesBuffer
    }

    func saveAsPlyFile(fileName: String,
                       beforeGlobalThread: [() -> Void],
                       afterGlobalThread: [() -> Void],
                       errorCallback: (XError) -> Void,
                       format: String) {

        guard !isSavingFile else {
            return errorCallback(XError.alreadySavingFile)
        }
        guard !cpuParticlesBuffer.isEmpty else {
            return errorCallback(XError.noScanDone)
        }

        DispatchQueue.global().async {
            self.isSavingFile = true
            DispatchQueue.main.async {
                for task in beforeGlobalThread { task() }
            }

            do { self.savedCloudURLs.append(try PLYFile.write(
                    fileName: fileName,
                    cpuParticlesBuffer: &self.cpuParticlesBuffer,
                    highConfCount: self.highConfCount,
                    format: format)) } catch {
                self.savingError = XError.savingFailed
            }

            DispatchQueue.main.async {
                for task in afterGlobalThread { task() }
            }
            self.isSavingFile = false
        }
    }

    func clearParticles() {
        highConfCount = 0
        currentPointIndex = 0
        currentPointCount = 0
        cpuParticlesBuffer = [CPUParticle]()
        rgbUniformsBuffers = [MetalBuffer<RGBUniforms>]()
        pointCloudUniformsBuffers = [MetalBuffer<PointCloudUniforms>]()

        commandQueue = device.makeCommandQueue()!
        for _ in 0 ..< maxInFlightBuffers {
            rgbUniformsBuffers.append(.init(device: device, count: 1, index: 0))
            pointCloudUniformsBuffers.append(.init(device: device, count: 1, index: kPointCloudUniforms.rawValue))
        }
        particlesBuffer = .init(device: device, count: maxPoints, index: kParticleUniforms.rawValue)
    }
    
    // Request to save the next depth frame as a photo
    func saveDepthPhoto() {
        shouldSaveNextDepth = true
        print("📸 Depth photo will be saved on next frame")
    }

    func loadSavedClouds() {
        let docs = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask)[0]
        savedCloudURLs = try! FileManager.default.contentsOfDirectory(
            at: docs, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)
    }

    // MARK: - Data Extraction for Gaussian Splatting
    func startDataExtraction() {
        isDataExtractionEnabled = true
        frameCounter = 0
        extractedFrameCounter = 0
        createDataExtractionDirectories()
        print("Started data extraction at \(dataExtractionFPS) FPS")
    }

    func stopDataExtraction() {
        guard isDataExtractionEnabled else { return }
        isDataExtractionEnabled = false
        exportCOLMAPFiles()
        print("Stopped data extraction. Total frames: \(extractedFrameCounter)")
    }

    private func createDataExtractionDirectories() {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        // Create folder with timestamp for this scan session
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = dateFormatter.string(from: Date())
        let scanFolderName = "scan_\(timestamp)"

        currentScanFolderName = scanFolderName

        let exportURL = documentsURL.appendingPathComponent(scanFolderName)
        let imagesURL = exportURL.appendingPathComponent("images")
        let sparseURL = exportURL.appendingPathComponent("sparse")
        let sparse0URL = sparseURL.appendingPathComponent("0")

        try? FileManager.default.createDirectory(at: exportURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: imagesURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: sparse0URL, withIntermediateDirectories: true)

        print("Created scan folder: \(scanFolderName)")
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
        // Convert ARFrame's captured image to JPEG
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()

        // Convert to RGB color space and create JPEG data
        if let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
           let jpegData = context.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: [:]) {

            let imageURL = baseURL.appendingPathComponent("images").appendingPathComponent(String(format: "%06d.jpg", frameIndex))

            do {
                try jpegData.write(to: imageURL)
            } catch {
                print("Failed to save image \(frameIndex): \(error)")
            }
        }
    }

    private func saveCameraIntrinsics(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let intrinsics = frame.camera.intrinsics
        let imageResolution = frame.camera.imageResolution

        // Extract intrinsic parameters
        let fx = intrinsics[0][0]  // Focal length X
        let fy = intrinsics[1][1]  // Focal length Y
        let cx = intrinsics[2][0]  // Principal point X
        let cy = intrinsics[2][1]  // Principal point Y

        // COLMAP cameras.txt format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
        // Using PINHOLE model: fx fy cx cy
        let cameraLine = "\(frameIndex) PINHOLE \(Int(imageResolution.width)) \(Int(imageResolution.height)) \(fx) \(fy) \(cx) \(cy)\n"

        let camerasURL = baseURL.appendingPathComponent("sparse/0/cameras.txt")

        // Append to cameras.txt file
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
        // Get camera transform (camera-to-world)
        let cameraToWorld = frame.camera.transform

        // Convert to world-to-camera (what COLMAP expects)
        let worldToCamera = cameraToWorld.inverse

        // Extract rotation (as quaternion) and translation
        let rotation = simd_quaternion(worldToCamera)
        let translation = worldToCamera.columns.3

        // COLMAP images.txt format: 
        // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        let imageLine = "\(frameIndex) \(rotation.real) \(rotation.imag.x) \(rotation.imag.y) \(rotation.imag.z) \(translation.x) \(translation.y) \(translation.z) \(frameIndex) \(String(format: "%06d.jpg", frameIndex))\n"

        let imagesURL = baseURL.appendingPathComponent("sparse/0/images.txt")

        // Append to images.txt file
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

        print("COLMAP data exported to: \(exportURL.path)")
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
                print("Exported \(pointID) high-confidence points to points3D.txt")
            } catch {
                print("Failed to write points3D.txt: \(error)")
            }
        }
    }
}

// MARK: - Metal Renderer Helpers
private extension Renderer {
    func makeUnprojectionPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "unprojectVertex") else {
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
        guard let vertexFunction = library.makeFunction(name: "rgbVertex"),
            let fragmentFunction = library.makeFunction(name: "rgbFragment") else {
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
        guard let vertexFunction = library.makeFunction(name: "particleVertex"),
            let fragmentFunction = library.makeFunction(name: "particleFragment") else {
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
    func makeGridPoints() -> [Float2] {
        // If we have cached probability samples from PromptDA LoG, use them
        if let cached = cachedProbSamples, !cached.isEmpty {
            print("📊 Using LoG probability samples: \(cached.count) points")
            return cached
        }
        
        // Fallback to uniform hexagonal grid (original behavior for LiDAR)
        print("📊 Using uniform hexagonal grid: ~\(numGridPoints) points")
        let gridArea = cameraResolution.x * cameraResolution.y
        let spacing = sqrt(gridArea / Float(numGridPoints))
        let deltaX = Int(round(cameraResolution.x / spacing))
        let deltaY = Int(round(cameraResolution.y / spacing))

        var points = [Float2]()
        for gridY in 0 ..< deltaY {
            let alternatingOffsetX = Float(gridY % 2) * spacing / 2
            for gridX in 0 ..< deltaX {
                let cameraPoint = Float2(alternatingOffsetX + (Float(gridX) + 0.5) * spacing, (Float(gridY) + 0.5) * spacing)

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
        return flipYZ * matrix_float4x4(simd_quaternion(rotationAngle, Float3(0, 0, 1)))
    }
}
