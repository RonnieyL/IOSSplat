//
//  Renderer.swift
//  SceneDepthPointCloud

import Metal
import MetalKit
import ARKit
import CoreImage


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
    var numGridPoints = 2_000
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
        guard let depthMap = frame.smoothedSceneDepth?.depthMap,
            let confidenceMap = frame.smoothedSceneDepth?.confidenceMap else {
                return false
        }
        
        depthTexture = makeTexture(fromPixelBuffer: depthMap, pixelFormat: .r32Float, planeIndex: 0)
        confidenceTexture = makeTexture(fromPixelBuffer: confidenceMap, pixelFormat: .r8Uint, planeIndex: 0)
        
        return true
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
        
        currentPointIndex = (currentPointIndex + gridPointsBuffer.count) % maxPoints
        currentPointCount = min(currentPointCount + gridPointsBuffer.count, maxPoints)
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
        
        // Create empty points3D.txt (or export sparse point cloud if desired)
        let pointsURL = exportURL.appendingPathComponent("sparse/0/points3D.txt")
        let emptyPointsData = "# 3D point list with one line of data per point:\n# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)\n".data(using: .utf8)
        try? emptyPointsData?.write(to: pointsURL)
        
        print("COLMAP data exported to: \(exportURL.path)")
        print("Total extracted frames: \(extractedFrameCounter)")
        print("Files accessible in iOS Files app under: \(currentScanFolderName)")
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
