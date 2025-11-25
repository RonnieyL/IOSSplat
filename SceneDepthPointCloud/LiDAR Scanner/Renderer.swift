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
    var cpuParticlesBuffer = [CPUParticle]()
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
    var frameCounter = 0
    var extractedFrameCounter = 0
    var currentScanFolderName = ""
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
                                                            array: makeGridPoints(frame: nil),
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
    
    // Depth photo saving
    var depthPhotoCounter = 0
    var shouldSaveNextDepth = false

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
        if depthSource == .mvs {
            do {
                // Use 256×192 prompt size (matches ARKit depth directly, no rotation)
                promptDAEngine = try PromptDAEngine.create(
                    bundleModelName: "PromptDA_vits_518x518_prompt256x192",
                    rgbSize: .init(width: 518, height: 518),
                    promptHW: (256, 192)
                )
            } catch {
                print("   Error: \(error.localizedDescription)")
                if let nsError = error as NSError? {
                    print("   Domain: \(nsError.domain)")
                    print("   Code: \(nsError.code)")
                    if let reason = nsError.userInfo[NSLocalizedFailureReasonErrorKey] as? String {
                        print("   Reason: \(reason)")
                    }
                }
            }
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

        depthTexture = makeTexture(fromPixelBuffer: depthMap, pixelFormat: .r32Float, planeIndex: 0)
        confidenceTexture = makeTexture(fromPixelBuffer: confidenceMap, pixelFormat: .r8Uint, planeIndex: 0)
        
        // Save depth photo if requested
        if shouldSaveNextDepth {
            saveDepthAsPhoto(depthMap)
            shouldSaveNextDepth = false
        }

        return true
    }
    
    private func updateMVSDepthTextures(frame: ARFrame) -> Bool {
        guard let promptDA = promptDAEngine else {
            print("⚠️ PromptDA not available, falling back to LiDAR")
            return updateLiDARDepthTextures(frame: frame)
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

            let depthW = CVPixelBufferGetWidth(output.depthPB)
            let depthH = CVPixelBufferGetHeight(output.depthPB)
            
            // Always use synthetic high confidence map for PromptDA points
            confidenceTexture = makeSyntheticConfidenceTexture(width: depthW, height: depthH)

            return true
            
        } catch {
            print("❌ PromptDA failed: \(error.localizedDescription), falling back to LiDAR")
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
        if depthSource == .mvs {
            let newPoints = makeGridPoints(frame: frame)
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
    func makeGridPoints(frame: ARFrame? = nil) -> [Float2] {
        // Custom Bernoulli sampling from probability buffer
        if let frame = frame, depthSource == .mvs, let promptDA = promptDAEngine {
            // Create CIImage from captured image
            let rgb = CIImage(cvPixelBuffer: frame.capturedImage)
            
            // Determine target size (use depth texture size if available, else default)
            var targetSize = CGSize(width: 518, height: 518)
            if let tex = depthTexture {
                targetSize = CGSize(width: CVMetalTextureGetTexture(tex)?.width ?? 518, 
                                  height: CVMetalTextureGetTexture(tex)?.height ?? 518)
            }
            
            do {
                // Generate probability map on the fly
                let probPB = try promptDA.makeNewLoGProbability(from: rgb, size: targetSize)
                
                CVPixelBufferLockBaseAddress(probPB, .readOnly)
                defer { CVPixelBufferUnlockBaseAddress(probPB, .readOnly) }
                
                let width = CVPixelBufferGetWidth(probPB)
                let height = CVPixelBufferGetHeight(probPB)
                let stride = CVPixelBufferGetBytesPerRow(probPB) / MemoryLayout<Float>.stride
                
                guard let ptr = CVPixelBufferGetBaseAddress(probPB)?.assumingMemoryBound(to: Float.self) else {
                    return []
                }
                
                var points = [Float2]()
                let scaleX = Float(cameraResolution.x) / Float(width)
                let scaleY = Float(cameraResolution.y) / Float(height)
                
                for y in 0..<height {
                    let row = y * stride
                    for x in 0..<width {
                        let prob = ptr[row + x]
                        if Float.random(in: 0...1) < prob {
                            let cx = Float(x) * scaleX
                            let cy = Float(y) * scaleY
                            points.append(Float2(cx, cy))
                        }
                    }
                }
                
                print("📊 Sampled \(points.count) points")
                return points
                
            } catch {
                print("⚠️ Failed to generate probability map: \(error)")
            }
        }
        
        // Fallback to uniform hexagonal grid (original behavior for LiDAR)
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
