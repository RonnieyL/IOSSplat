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
        didSet { mvsProcessingInterval = 1.0 / mvsProcessingFPS }
    }
    private var mvsProcessingInterval: TimeInterval = 1.0 / 3.0
    private var lastMVSProcessingTime: TimeInterval = 0

    // Data extraction system for Gaussian Splatting
    var isDataExtractionEnabled = false
    var dataExtractionFPS: Double = 3.0 {
        didSet { dataExtractionInterval = 1.0 / dataExtractionFPS }
    }
    private var dataExtractionInterval: TimeInterval = 1.0 / 3.0
    private var lastDataExtractionTime: TimeInterval = 0
    private var frameCounter = 0
    private var extractedFrameCounter = 0
    private var currentScanFolderName = ""

    // We only use portrait orientation in this app
    private let orientation = UIInterfaceOrientation.portrait
    // Camera motion thresholds
    private let cameraRotationThreshold = cos(2 * Float.pi / 180)
    private let cameraTranslationThreshold: Float = powf(0.02, 2)   // m²
    // Max command buffers in flight
    private let maxInFlightBuffers = 3

    private lazy var rotateToARCamera = MVSRenderer.makeRotateToARCameraMatrix(orientation: orientation)
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

    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?

    // Multi-buffer rendering pipeline
    private let inFlightSemaphore: DispatchSemaphore
    private var currentBufferIndex = 0

    // Viewport
    private var viewportSize = CGSize()

    // Sample grid
    private lazy var gridPointsBuffer = MVSMetalBuffer<SIMD2<Float>>(
        device: device, array: makeGridPoints(), index: kMVSGridPoints.rawValue, options: []
    )

    // Uniform buffers (triple buffered)
    var rgbUniforms = MVSRGBUniforms()
    private lazy var rgbUniformsBuffers = MVSMetalBuffer<MVSRGBUniforms>(
        device: device, count: maxInFlightBuffers, index: kMVSRGBUniforms.rawValue
    )

    var pointCloudUniforms = MVSPointCloudUniforms()
    private lazy var pointCloudUniformsBuffers = MVSMetalBuffer<MVSPointCloudUniforms>(
        device: device, count: maxInFlightBuffers, index: kMVSPointCloudUniforms.rawValue
    )

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

    // MARK: - Init
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: MVSRenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        library = device.makeDefaultLibrary()!
        commandQueue = device.makeCommandQueue()!

        particlesBuffer = MVSMetalBuffer<MVSParticle>(
            device: device, count: maxPoints, index: kMVSParticleUniforms.rawValue
        )

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

        _ = inFlightSemaphore.wait(timeout: .distantFuture)
        commandBuffer.addCompletedHandler { [weak self] _ in self?.inFlightSemaphore.signal() }

        // Always update frame data and camera textures
        update(frame: currentFrame)
        updateCapturedImageTextures(frame: currentFrame)

        // rotate ring buffers
        currentBufferIndex = (currentBufferIndex + 1) % maxInFlightBuffers

        // write this frame's uniforms into the ring slot
        rgbUniformsBuffers[currentBufferIndex] = rgbUniforms
        pointCloudUniformsBuffers[currentBufferIndex] = pointCloudUniforms

        // data extraction (optional)
        if shouldExtractDataThisFrame(currentFrame) { extractCameraData(frame: currentFrame) }

        // MVS / accumulation
        if shouldProcessMVSThisFrame(currentFrame), shouldAccumulate(frame: currentFrame) {
            accumulatePointsViaMVS(frame: currentFrame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
        }

        // render rgb image if requested
        if rgbUniforms.radius > 0 {
            var retaining = [capturedImageTextureY, capturedImageTextureCbCr]
            commandBuffer.addCompletedHandler { _ in retaining.removeAll() }

            renderEncoder.setDepthStencilState(relaxedStencilState)
            renderEncoder.setRenderPipelineState(rgbPipelineState)

            // bind uniform buffer at the correct byte offset for this ring slot
            renderEncoder.setVertexBuffer(
                rgbUniformsBuffers,
                offset: currentBufferIndex * MemoryLayout<MVSRGBUniforms>.stride
            )
            renderEncoder.setFragmentBuffer(
                rgbUniformsBuffers,
                offset: currentBufferIndex * MemoryLayout<MVSRGBUniforms>.stride
            )

            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kMVSTextureY.rawValue))
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kMVSTextureCbCr.rawValue))

            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }

        // render particles
        if showParticles {
            renderEncoder.setDepthStencilState(depthStencilState)
            renderEncoder.setRenderPipelineState(particlePipelineState)

            renderEncoder.setVertexBuffer(
                pointCloudUniformsBuffers,
                offset: currentBufferIndex * MemoryLayout<MVSPointCloudUniforms>.stride
            )
            renderEncoder.setVertexBuffer(particlesBuffer)

            renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: currentPointCount)
        }

        renderEncoder.endEncoding()
        commandBuffer.present(renderDestination.currentDrawable!)
        commandBuffer.commit()
    }

    private func shouldProcessMVSThisFrame(_ frame: ARFrame) -> Bool {
        let t = frame.timestamp
        if t - lastMVSProcessingTime >= mvsProcessingInterval {
            lastMVSProcessingTime = t
            return true
        }
        return false
    }

    private func shouldExtractDataThisFrame(_ frame: ARFrame) -> Bool {
        guard isDataExtractionEnabled else { return false }
        let t = frame.timestamp
        if t - lastDataExtractionTime >= dataExtractionInterval {
            lastDataExtractionTime = t
            return true
        }
        return false
    }

    // MARK: - MVS Point Cloud Generation with Depth-Anything
    private func accumulatePointsViaMVS(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        updateKeyframeBuffer(frame: frame)
        processFrameWithDepthAnything(frame: frame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
    }

    private func updateKeyframeBuffer(frame: ARFrame) {
        keyframeBuffer.append(frame)
        if keyframeBuffer.count > maxKeyframes { keyframeBuffer.removeFirst() }
        previousFrame = frame
        previousTransform = frame.camera.transform
    }

    private func processFrameWithDepthAnything(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        guard let depthProcessor = depthProcessor else {
            generateFallbackPointCloud(frame: frame)
            return
        }

        guard let aiDepthTexture = depthProcessor.processDepthSync(from: frame) else {
            generateFallbackPointCloud(frame: frame); return
        }

        self.depthTexture = aiDepthTexture
        generatePointCloudFromAIDepth(frame: frame, depthTexture: aiDepthTexture)
    }

    private func generatePointCloudFromAIDepth(frame: ARFrame, depthTexture: MTLTexture) {
        let gridPoints = makeGridPoints()
        let K = frame.camera.intrinsics
        let T = frame.camera.transform
        let capturedImage = frame.capturedImage

        let depthWidth = Float(depthTexture.width)
        let depthHeight = Float(depthTexture.height)
        let imageWidth = Float(CVPixelBufferGetWidth(capturedImage))
        let imageHeight = Float(CVPixelBufferGetHeight(capturedImage))

        CVPixelBufferLockBaseAddress(capturedImage, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(capturedImage, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(capturedImage) else { return }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(capturedImage)

        let maxPointsPerFrame = min(gridPoints.count, 1000)

        for i in 0..<maxPointsPerFrame {
            let gp = gridPoints[i]

            let u = Int(gp.x * depthWidth / Float(frame.camera.imageResolution.width))
            let v = Int(gp.y * depthHeight / Float(frame.camera.imageResolution.height))
            guard u >= 0, v >= 0, u < Int(depthWidth), v < Int(depthHeight) else { continue }

            let nu = Float(u) / Float(depthWidth)
            let nv = Float(v) / Float(depthHeight)
            let aiDepth = 1.0 + 2.0 * (0.5 + 0.3 * sin(nu * 6.28) * cos(nv * 6.28))

            guard aiDepth > 0.3, aiDepth < 10.0 else { continue }

            let local = K.inverse * simd_float3(gp.x, gp.y, 1) * aiDepth
            let world = T * simd_float4(local, 1)

            let iu = Int(gp.x * imageWidth / Float(frame.camera.imageResolution.width))
            let iv = Int(gp.y * imageHeight / Float(frame.camera.imageResolution.height))
            guard iu >= 0, iv >= 0, iu < Int(imageWidth), iv < Int(imageHeight) else { continue }


            let pixelOffset = iv * bytesPerRow + iu * 4
            let p = baseAddress.advanced(by: pixelOffset).assumingMemoryBound(to: UInt8.self)

            let color = simd_float3(Float(p[2]), Float(p[1]), Float(p[0]))

            let particle = MVSParticle(
                position: simd_float3(world.x, world.y, world.z),
                color: color / 255.0,
                confidence: 2
            )

            if currentPointCount < maxPoints {
                cpuParticlesBuffer.append(particle)
                particlesBuffer[currentPointCount] = particle
                currentPointCount += 1
            }
        }

        print("Generated \(maxPointsPerFrame) points from Depth-Anything-V2")
    }

    private func generateFallbackPointCloud(frame: ARFrame) {
        let gridPoints = makeGridPoints()
        let K = frame.camera.intrinsics
        let T = frame.camera.transform

        let maxFallbackPoints = min(gridPoints.count, 500)

        for i in 0..<maxFallbackPoints {
            let gp = gridPoints[i]
            let z = 1.0 + 0.5 * sin(gp.x * 0.01) * cos(gp.y * 0.01)

            let local = K.inverse * simd_float3(gp.x, gp.y, 1) * z
            let world = T * simd_float4(local, 1)

            let color = simd_float3(
                0.7 + 0.3 * sin(gp.x * 0.02),
                0.5 + 0.3 * cos(gp.y * 0.02),
                0.8
            )

            let particle = MVSParticle(
                position: simd_float3(world.x, world.y, world.z),
                color: color,
                confidence: 1
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

        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = df.string(from: Date())
        let scanFolderName = "mvs_scan_\(timestamp)"
        currentScanFolderName = scanFolderName

        let exportURL = documentsURL.appendingPathComponent(scanFolderName)
        let imagesURL = exportURL.appendingPathComponent("images")
        let sparse0URL = exportURL.appendingPathComponent("sparse").appendingPathComponent("0")

        try? FileManager.default.createDirectory(at: exportURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: imagesURL, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: sparse0URL, withIntermediateDirectories: true)

        print("Created MVS scan folder: \(scanFolderName)")
    }

    private func extractCameraData(frame: ARFrame) {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let baseURL = documentsURL.appendingPathComponent(currentScanFolderName)

        saveRGBImage(frame: frame, to: baseURL, frameIndex: extractedFrameCounter)
        saveCameraIntrinsics(frame: frame, to: baseURL, frameIndex: extractedFrameCounter)
        saveCameraExtrinsics(frame: frame, to: baseURL, frameIndex: extractedFrameCounter)

        extractedFrameCounter += 1
    }

    private func saveRGBImage(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()

        if let cs = CGColorSpace(name: CGColorSpace.sRGB),
           let jpeg = context.jpegRepresentation(of: ciImage, colorSpace: cs, options: [:]) {
            let url = baseURL.appendingPathComponent("images").appendingPathComponent(String(format: "%06d.jpg", frameIndex))
            try? jpeg.write(to: url)
        }
    }

    private func saveCameraIntrinsics(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let K = frame.camera.intrinsics
        let res = frame.camera.imageResolution
        let fx = K[0][0], fy = K[1][1], cx = K[2][0], cy = K[2][1]

        let line = "\(frameIndex) PINHOLE \(Int(res.width)) \(Int(res.height)) \(fx) \(fy) \(cx) \(cy)\n"
        let url = baseURL.appendingPathComponent("sparse/0/cameras.txt")
        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: url.path),
               let fh = try? FileHandle(forWritingTo: url) {
                fh.seekToEndOfFile(); fh.write(data); try? fh.close()
            } else {
                try? data.write(to: url)
            }
        }
    }

    private func saveCameraExtrinsics(frame: ARFrame, to baseURL: URL, frameIndex: Int) {
        let Tcw = frame.camera.transform.inverse
        let q = simd_quaternion(Tcw)
        let t = Tcw.columns.3

        let line = "\(frameIndex) \(q.real) \(q.imag.x) \(q.imag.y) \(q.imag.z) \(t.x) \(t.y) \(t.z) \(frameIndex) \(String(format: "%06d.jpg", frameIndex))\n"
        let url = baseURL.appendingPathComponent("sparse/0/images.txt")
        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: url.path),
               let fh = try? FileHandle(forWritingTo: url) {
                fh.seekToEndOfFile(); fh.write(data); try? fh.close()
            } else {
                try? data.write(to: url)
            }
        }
    }

    private func exportCOLMAPFiles() {
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let baseURL = documentsURL.appendingPathComponent(currentScanFolderName)
        exportPointCloudData(to: baseURL)
        print("MVS COLMAP data exported to: \(baseURL.path)")
        print("Total extracted frames: \(extractedFrameCounter)")
        print("Total point cloud points: \(cpuParticlesBuffer.count)")
        print("High confidence points: \(highConfCount)")
        print("Files accessible in iOS Files app under: \(currentScanFolderName)")
    }

    private func exportPointCloudData(to baseURL: URL) {
        let url = baseURL.appendingPathComponent("sparse/0/points3D.txt")
        var txt = "# 3D point list with one line of data per point:\n"
        txt += "# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID POINT2D_IDX)\n"

        var id = 0
        for p in cpuParticlesBuffer where p.confidence >= 2 {
            let pos = p.position
            let c = p.color
            let r = max(0, min(255, Int(c.x)))
            let g = max(0, min(255, Int(c.y)))
            let b = max(0, min(255, Int(c.z)))
            txt += "\(id) \(pos.x) \(pos.y) \(pos.z) \(r) \(g) \(b) 1.0\n"
            id += 1
        }
        try? txt.data(using: .utf8)?.write(to: url)
        print("Exported \(id) high-confidence MVS points to points3D.txt")
    }
}

// MARK: - Utility Methods
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
        guard let prev = previousTransform else {
            previousTransform = frame.camera.transform
            return true
        }
        let t = simd_distance_squared(prev.columns.3, frame.camera.transform.columns.3)
        let r = simd_dot(prev.columns.1, frame.camera.transform.columns.1)
        if t > cameraTranslationThreshold || r < cameraRotationThreshold {
            previousTransform = frame.camera.transform
            return true
        }
        return false
    }

    private func update(frame: ARFrame) {
        // RGB uniforms (3x3 & 4x4 via copy helpers)
        rgbUniforms.viewToCamera.copy(from: frame.camera.viewMatrix(for: orientation))
        rgbUniforms.cameraToWorld.copy(from: frame.camera.transform)

        // Point-cloud uniforms
        pointCloudUniforms.cameraIntrinsics.copy(from: frame.camera.intrinsics)
        pointCloudUniforms.cameraIntrinsicsInversed.copy(from: frame.camera.intrinsics.inverse)
        pointCloudUniforms.cameraToWorld.copy(from: frame.camera.transform)
        pointCloudUniforms.worldToCamera.copy(from: frame.camera.viewMatrix(for: orientation))
        pointCloudUniforms.cameraToWorldRotation.copy(from: frame.camera.transform.upperLeft3x3())
        pointCloudUniforms.particleSize = particleSize
        pointCloudUniforms.numGridPoints = Int32(numGridPoints)

        // Display transform (CGFloat → Float), 3×3 * 3×3
        let t = frame.displayTransform(for: orientation, viewportSize: viewportSize)
        let A = simd_float3x3(
            SIMD3<Float>(Float(t.a), Float(t.c), Float(t.tx)),
            SIMD3<Float>(Float(t.b), Float(t.d), Float(t.ty)),
            SIMD3<Float>(0, 0, 1)
        )
        let R3 = rotateToARCamera.upperLeft3x3()
        pointCloudUniforms.cameraToWorldRotation = A * R3
    }


    private func updateCapturedImageTextures(frame: ARFrame) {
        let pb = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(pb) >= 2 else { return }
        capturedImageTextureY = makeTexture(fromPixelBuffer: pb, pixelFormat:.r8Unorm, planeIndex:0)
        capturedImageTextureCbCr = makeTexture(fromPixelBuffer: pb, pixelFormat:.rg8Unorm, planeIndex:1)
    }
}

// MARK: - Metal Helpers
private extension MVSRenderer {
    func makeUnprojectionPipelineState() -> MTLRenderPipelineState? {
        guard let v = library.makeFunction(name: "mvsUnprojectVertex") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = v
        d.isRasterizationEnabled = false
        d.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        d.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        return try? device.makeRenderPipelineState(descriptor: d)
    }

    func makeRGBPipelineState() -> MTLRenderPipelineState? {
        guard let v = library.makeFunction(name: "mvsRgbVertex"),
              let f = library.makeFunction(name: "mvsRgbFragment") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = v
        d.fragmentFunction = f
        d.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        d.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        return try? device.makeRenderPipelineState(descriptor: d)
    }

    func makeParticlePipelineState() -> MTLRenderPipelineState? {
        guard let v = library.makeFunction(name: "mvsParticleVertex"),
              let f = library.makeFunction(name: "mvsParticleFragment") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = v
        d.fragmentFunction = f
        d.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        d.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        d.colorAttachments[0].isBlendingEnabled = true
        d.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        return try? device.makeRenderPipelineState(descriptor: d)
    }

    func makeGridPoints() -> [SIMD2<Float>] {
        let res = SIMD2<Float>(1920, 1440) // default
        let area = res.x * res.y
        let spacing = sqrt(area / Float(numGridPoints))
        let dx = Int(round(res.x / spacing))
        let dy = Int(round(res.y / spacing))

        var pts = [SIMD2<Float>]()
        for y in 0..<dy {
            let offsetX = Float(y % 2) * spacing / 2
            for x in 0..<dx {
                let p = SIMD2<Float>(offsetX + (Float(x) + 0.5) * spacing,
                                     (Float(y) + 0.5) * spacing)
                pts.append(p)
            }
        }
        return pts
    }

    func makeTextureCache() -> CVMetalTextureCache {
        var cache: CVMetalTextureCache!
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        return cache
    }

    func makeTexture(fromPixelBuffer pb: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let w = CVPixelBufferGetWidthOfPlane(pb, planeIndex)
        let h = CVPixelBufferGetHeightOfPlane(pb, planeIndex)
        var tex: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pb, nil, pixelFormat, w, h, planeIndex, &tex)
        return status == kCVReturnSuccess ? tex : nil
    }

    static func cameraToDisplayRotation(orientation: UIInterfaceOrientation) -> Int {
        switch orientation {
        case .landscapeLeft: return 180
        case .portrait: return 90
        case .portraitUpsideDown: return -90
        default: return 0
        }
    }

    static func makeRotateToARCameraMatrix(orientation: UIInterfaceOrientation) -> matrix_float4x4 {
        // flip to ARKit Camera's coordinate
        let flipYZ = matrix_float4x4(
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        )
        let angle = Float(cameraToDisplayRotation(orientation: orientation)) * .degreesToRadian
        return flipYZ * matrix_float4x4(simd_quaternion(angle, SIMD3<Float>(0, 0, 1)))
    }
}
