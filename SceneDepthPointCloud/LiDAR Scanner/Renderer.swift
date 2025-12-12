//
//  Renderer.swift
//  SceneDepthPointCloud

import Metal
import MetalKit
import ARKit
import CoreImage
import Foundation
import Photos

// MARK: - GPU Sampling Structures

struct SamplingUniforms {
    var width: UInt32
    var height: UInt32
    var stride: UInt32
    var scaleX: Float
    var scaleY: Float
    var maxPoints: UInt32
}

enum ComputeBufferIndices: Int {
    case kProbabilityMap = 0
    case kSampledPoints = 1
    case kAtomicCounter = 2
    case kSamplingUniforms = 3
}


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
    private lazy var samplingComputePipelineState = makeSamplingComputePipelineState()!
    // texture cache for captured image
    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?
    private var depthTexture: CVMetalTexture?
    private var confidenceTexture: CVMetalTexture?
    
    // GPU sampling buffers
    var sampledPointsBuffer: MTLBuffer?
    var atomicCounterBuffer: MTLBuffer?
    var probabilityBuffer: MTLBuffer?
    
    // Gaussian Splatting - MetalSplatter (old, keeping for fallback)
    private var gaussianSplatRenderer: GaussianSplatRenderer?

    // OpenSplat Renderer (new, advanced tile-based)
    private var openSplatRenderer: OpenSplatRenderer?
    var isGaussianSplattingEnabled = true  // Enable by default for testing
    var useOpenSplatRenderer = false  // Toggle for migration - DISABLED (using MetalSplatter)

    // Debug tracking for splatting pipeline
    private var splatRenderFrameCount: Int = 0
    private var splatUpdateFrameCount: Int = 0
    private var lastSplatDebugTime: TimeInterval = 0

    // OpenSplat Gaussian data buffers
    private var gaussianMeansBuffer: MTLBuffer?      // SIMD3<Float> positions [N]
    private var gaussianScalesBuffer: MTLBuffer?     // SIMD3<Float> scale per gaussian [N]
    private var gaussianQuatsBuffer: MTLBuffer?      // SIMD4<Float> rotation quaternion [N]
    private var gaussianColorsBuffer: MTLBuffer?     // SIMD3<Float> RGB [N]
    private var gaussianOpacitiesBuffer: MTLBuffer?  // Float opacity [N]
    private var currentGaussianCount: Int = 0
    private let maxGaussians = 500000  // Maximum Gaussians to store

    // Debug: Track buffer health
    private var bufferAllocationSuccessful = false
    
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
        self.library = device.makeDefaultLibrary()!
        self.commandQueue = device.makeCommandQueue()!
        
        // Initialize Gaussian Splatting
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✨ Initializing Gaussian Splatting Renderer")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.gaussianSplatRenderer = GaussianSplatRenderer(device: device)
        if gaussianSplatRenderer != nil {
            print("✅ GaussianSplatRenderer (MetalSplatter) initialized")
            print("   • Mode: Tile-based splatting (simple)")
            print("   • Status: ENABLED (fallback)")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        } else {
            print("❌ GaussianSplatRenderer initialization FAILED")
            print("   • Falling back to point cloud rendering")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        }

        // Initialize OpenSplat Renderer (Step 2 & 3)
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("🎯 STEP 2 & 3: Initializing OpenSplatRenderer")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        do {
            let startTime = CACurrentMediaTime()
            self.openSplatRenderer = try OpenSplatRenderer(device: device)
            let elapsed = CACurrentMediaTime() - startTime

            print("✅ OpenSplatRenderer initialized successfully")
            print("   • Init time: \(String(format: "%.3f", elapsed))s")
            print("   • Mode: Tile-based GPU rasterization")
            print("   • Features: 16x16 tiles, depth sorting, SH support")
            print("   • Status: INITIALIZED (inactive until Step 7)")

            // Validation: Check that pipeline states were created
            if self.openSplatRenderer != nil {
                print("   ✓ Renderer instance created")
                print("   ✓ Metal pipelines loaded")
                print("   ✓ GPU utilities initialized")
            }
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        } catch {
            print("❌ OpenSplatRenderer initialization FAILED")
            print("   • Error: \(error.localizedDescription)")
            if let nsError = error as NSError? {
                print("   • Domain: \(nsError.domain)")
                print("   • Code: \(nsError.code)")
                if let reason = nsError.userInfo[NSLocalizedFailureReasonErrorKey] as? String {
                    print("   • Reason: \(reason)")
                }
            }
            print("   • Fallback: Will use MetalSplatter instead")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        }

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
        self.loadSavedClouds()

        // Initialize OpenSplat Gaussian buffers (after all properties initialized)
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("📦 STEP 1: Allocating OpenSplat Gaussian Buffers")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        bufferAllocationSuccessful = allocateGaussianBuffers()
        if bufferAllocationSuccessful {
            print("✅ All Gaussian buffers allocated successfully")
            validateGaussianBuffers()
        } else {
            print("❌ Gaussian buffer allocation FAILED")
        }
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
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

        // Gaussian Splatting
        if isGaussianSplattingEnabled, let splatting = gaussianSplatRenderer {
            // Check if we have Gaussian splats to render
            // For now, always use the Gaussian path when enabled
            // The GaussianSplatRenderer will handle the case of no points gracefully
            renderEncoder.endEncoding()

            splatRenderFrameCount += 1
            let currentTime = CACurrentMediaTime()
            let shouldPrintDebug = (currentTime - lastSplatDebugTime) >= 2.0 // Debug every 2 seconds

            if shouldPrintDebug {
                print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print("🎨 GAUSSIAN SPLATTING PIPELINE DEBUG")
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print("📊 Stats:")
                print("   • Total splats: \(splatting.splatCount)")
                print("   • Render frames: \(splatRenderFrameCount)")
                print("   • Update frames: \(splatUpdateFrameCount)")
                print("   • Viewport: \(Int(viewportSize.width))×\(Int(viewportSize.height))")
                lastSplatDebugTime = currentTime
            }

            if let drawable = renderDestination.currentDrawable {
                // Use viewMatrix(for: orientation) to properly handle portrait mode
                let viewMatrix = currentFrame.camera.viewMatrix(for: orientation)
                let projectionMatrix = currentFrame.camera.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: 0.001, zFar: 1000)

                if shouldPrintDebug {
                    print("🎥 Camera:")
                    print("   • Position: (\(String(format: "%.2f", currentFrame.camera.transform.columns.3.x)), \(String(format: "%.2f", currentFrame.camera.transform.columns.3.y)), \(String(format: "%.2f", currentFrame.camera.transform.columns.3.z)))")
                    print("   • Orientation: \(orientation)")
                    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
                }

                // STEP 7A: Try OpenSplat rendering (to temp texture, not displayed yet)
                if self.useOpenSplatRenderer,
                   let openSplat = self.openSplatRenderer,
                   let meansBuffer = self.gaussianMeansBuffer,
                   let scalesBuffer = self.gaussianScalesBuffer,
                   let quatsBuffer = self.gaussianQuatsBuffer,
                   let colorsBuffer = self.gaussianColorsBuffer,
                   let opacitiesBuffer = self.gaussianOpacitiesBuffer,
                   self.currentGaussianCount > 0 {

                    print("🎬 [Step 7A] Attempting OpenSplat render...")
                    print("   • Gaussian count: \(self.currentGaussianCount)")

                    // Extract camera parameters
                    let camParams = extractCameraParameters(from: currentFrame)
                    print("   • Camera: fx=\(String(format: "%.1f", camParams.fx)), fy=\(String(format: "%.1f", camParams.fy))")
                    print("   • Image size: \(camParams.imgWidth)×\(camParams.imgHeight)")

                    // Create temporary output texture (same format as drawable)
                    let outputDesc = MTLTextureDescriptor.texture2DDescriptor(
                        pixelFormat: drawable.texture.pixelFormat,
                        width: drawable.texture.width,
                        height: drawable.texture.height,
                        mipmapped: false
                    )
                    outputDesc.usage = [.renderTarget, .shaderRead]
                    outputDesc.storageMode = .private

                    if let tempTexture = self.device.makeTexture(descriptor: outputDesc) {
                        print("   • Temp texture created: \(tempTexture.width)×\(tempTexture.height)")

                        // DEBUG: Check Gaussian positions before rendering
                        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: self.currentGaussianCount)
                        let colorsPtr = colorsBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: self.currentGaussianCount)

                        print("   🔍 DEBUG - First 5 Gaussians:")
                        for i in 0..<min(5, self.currentGaussianCount) {
                            let pos = meansPtr[i]
                            let col = colorsPtr[i]
                            print("      [\(i)] pos=(\(String(format: "%.3f", pos.x)), \(String(format: "%.3f", pos.y)), \(String(format: "%.3f", pos.z))), color=(\(String(format: "%.2f", col.x)), \(String(format: "%.2f", col.y)), \(String(format: "%.2f", col.z)))")
                        }

                        // Check camera position
                        let camPos = currentFrame.camera.transform.columns.3
                        print("   🔍 Camera position: (\(String(format: "%.3f", camPos.x)), \(String(format: "%.3f", camPos.y)), \(String(format: "%.3f", camPos.z)))")

                        // Calculate distance to first Gaussian
                        if self.currentGaussianCount > 0 {
                            let firstPos = meansPtr[0]
                            let dist = simd_distance(SIMD3<Float>(camPos.x, camPos.y, camPos.z), firstPos)
                            print("   🔍 Distance to first Gaussian: \(String(format: "%.3f", dist))m")
                            
                            // Verify view space depth with BOTH matrices
                            let viewMatrix = camParams.viewMatrix
                            let viewMatrixTransposed = camParams.viewMatrix.transpose
                            let firstGaussianHomo = SIMD4<Float>(firstPos.x, firstPos.y, firstPos.z, 1.0)
                            let viewSpace = viewMatrix * firstGaussianHomo
                            let viewSpaceTransposed = viewMatrixTransposed * firstGaussianHomo

                            print("   🔍 View Matrix [row 2]: [\(String(format: "%.3f", viewMatrix[0][2])), \(String(format: "%.3f", viewMatrix[1][2])), \(String(format: "%.3f", viewMatrix[2][2])), \(String(format: "%.3f", viewMatrix[3][2]))]")
                            print("   🔍 View space Z with original matrix: \(String(format: "%.3f", viewSpace.z))")
                            print("   🔍 View space Z with transposed matrix: \(String(format: "%.3f", viewSpaceTransposed.z))")
                            print("   🔍 View space FULL with original: (\(String(format: "%.3f", viewSpace.x)), \(String(format: "%.3f", viewSpace.y)), \(String(format: "%.3f", viewSpace.z)))")
                            print("   🔍 Expected: negative Z for visible Gaussians in ARKit")
                        }

                        do {
                            let renderStart = CACurrentMediaTime()

                            try openSplat.render(
                                gaussianMeans: meansBuffer,
                                gaussianScales: scalesBuffer,
                                gaussianQuats: quatsBuffer,
                                gaussianColors: colorsBuffer,
                                gaussianOpacities: opacitiesBuffer,
                                numGaussians: self.currentGaussianCount,
                                viewMatrix: camParams.viewMatrix,
                                projMatrix: camParams.projMatrix,
                                fx: camParams.fx,
                                fy: camParams.fy,
                                cx: camParams.cx,
                                cy: camParams.cy,
                                imgWidth: camParams.imgWidth,
                                imgHeight: camParams.imgHeight,
                                outputTexture: tempTexture,
                                globalScale: 1.0,
                                clipThreshold: -0.01,  // Clip if Z >= -0.01 (within 1cm or behind camera)
                                background: SIMD3<Float>(0, 0, 0)
                            )

                            let renderTime = CACurrentMediaTime() - renderStart
                            print("   ✅ OpenSplat render SUCCESS!")
                            print("   • Render time: \(String(format: "%.3f", renderTime))s (\(String(format: "%.1f", 1.0/renderTime)) FPS)")
                            print("   • Output texture: \(tempTexture.width)×\(tempTexture.height)")
                            print("   ⚠️ NOTE: Rendering to temp texture, NOT displayed yet (Step 7B)")

                        } catch {
                            print("   ❌ OpenSplat render FAILED: \(error.localizedDescription)")
                        }
                    } else {
                        print("   ❌ Failed to create temp output texture")
                    }
                } else if self.useOpenSplatRenderer {
                    print("   ⚠️ OpenSplat skipped: count=\(self.currentGaussianCount), renderer=\(self.openSplatRenderer != nil)")
                }

                // Still render with MetalSplatter for now (fallback/comparison)
                // DEBUG: Test MetalSplatter's view matrix transformation
                if shouldPrintDebug && self.currentGaussianCount > 0, let meansBuffer = self.gaussianMeansBuffer {
                    let meansPtr = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)
                    let firstPos = meansPtr[0]
                    let firstGaussianHomo = SIMD4<Float>(firstPos.x, firstPos.y, firstPos.z, 1.0)
                    let metalSplatterViewSpace = viewMatrix * firstGaussianHomo
                    print("   🔍 MetalSplatter view space Z: \(String(format: "%.3f", metalSplatterViewSpace.z))")
                    print("   🔍 MetalSplatter uses SAME viewMatrix as OpenSplat")
                }

                // DEBUG: Check for size mismatch between viewport and texture
                if shouldPrintDebug {
                    print("🔍 VIEWPORT DEBUG:")
                    print("   • viewportSize: \(Int(viewportSize.width))×\(Int(viewportSize.height))")
                    print("   • drawable.texture size: \(drawable.texture.width)×\(drawable.texture.height)")
                    if Int(viewportSize.width) != drawable.texture.width || Int(viewportSize.height) != drawable.texture.height {
                        print("   ⚠️ SIZE MISMATCH! This causes projection errors!")
                    }

                    // Check actual Gaussian positions and their projections
                    if let meansBuffer = self.gaussianMeansBuffer, self.currentGaussianCount > 0 {
                        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)
                        print("\n🔍 GAUSSIAN POSITION DEBUG (first 5):")
                        for i in 0..<min(5, self.currentGaussianCount) {
                            let worldPos = meansPtr[i]
                            let worldPosHomo = SIMD4<Float>(worldPos.x, worldPos.y, worldPos.z, 1.0)

                            // Transform to view space
                            let viewPos = viewMatrix * worldPosHomo

                            // Project to clip space
                            let clipPos = projectionMatrix * viewPos

                            // Normalize to NDC (-1 to 1)
                            let ndc = SIMD3<Float>(clipPos.x / clipPos.w, clipPos.y / clipPos.w, clipPos.z / clipPos.w)

                            // Convert to screen space (0 to width/height)
                            let screenX = (ndc.x + 1.0) * 0.5 * Float(viewportSize.width)
                            let screenY = (1.0 - ndc.y) * 0.5 * Float(viewportSize.height)  // Flip Y for screen coords

                            print("   [\(i)] world: (\(String(format: "%.2f", worldPos.x)), \(String(format: "%.2f", worldPos.y)), \(String(format: "%.2f", worldPos.z)))")
                            print("       view: Z=\(String(format: "%.2f", viewPos.z)) | screen: (\(String(format: "%.0f", screenX)), \(String(format: "%.0f", screenY)))")
                        }

                        // Check Y distribution of all Gaussians
                        var yMin: Float = .infinity
                        var yMax: Float = -.infinity
                        var ySum: Float = 0
                        for i in 0..<self.currentGaussianCount {
                            let y = meansPtr[i].y
                            yMin = min(yMin, y)
                            yMax = max(yMax, y)
                            ySum += y
                        }
                        let yAvg = ySum / Float(self.currentGaussianCount)
                        print("\n   📊 Y-coordinate distribution:")
                        print("      • Min: \(String(format: "%.2f", yMin)), Max: \(String(format: "%.2f", yMax)), Avg: \(String(format: "%.2f", yAvg))")
                        print("      • Camera Y: \(String(format: "%.2f", currentFrame.camera.transform.columns.3.y))")
                    }
                }

                splatting.draw(commandBuffer: commandBuffer,
                               viewMatrix: viewMatrix,
                               projectionMatrix: projectionMatrix,
                               viewport: viewportSize,
                               outputTexture: drawable.texture)
            }

            commandBuffer.present(renderDestination.currentDrawable!)
            commandBuffer.commit()
        } else {
            // render particles (fallback when Gaussian Splatting disabled)
            if self.showParticles {
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
        
        // Regenerate grid points buffer if we have new probability samples (MVS mode)
        if depthSource == .mvs || promptDAEngine != nil {
            let newPoints = makeGridPoints(frame: frame)
            print("   • Regenerating grid points buffer: \(newPoints.count) LoG-sampled points")
            gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                                  array: newPoints,
                                                  index: kGridPoints.rawValue, options: [])
        }
        
        // Capture values BEFORE GPU work for the completion handler
        let startIndex = currentPointIndex
        let expectedCount = gridPointsBuffer.count

        commandBuffer.addCompletedHandler { [weak self] buffer in
            guard let self = self else { return }
            retainingTextures.removeAll()
            
            // Read points from the known range that was just written
            var newPoints: [SIMD3<Float>] = []
            var newColors: [SIMD3<Float>] = []
            
            for offset in 0..<expectedCount {
                let i = (startIndex + offset) % self.maxPoints
                let position = self.particlesBuffer[i].position
                let color = self.particlesBuffer[i].color
                let confidence = self.particlesBuffer[i].confidence

                // Skip zero positions (invalid points)
                if position == simd_float3(0.0, 0.0, 0.0) {
                    continue
                }

                if confidence == 2 { self.highConfCount += 1 }
                self.cpuParticlesBuffer.append(
                    CPUParticle(position: position,
                                color: color,
                                confidence: confidence))

                // DEBUG: Check position values during capture
                if offset < 3 && newPoints.isEmpty, let frame = self.session.currentFrame {
                    print("🔍 Captured point[\(offset)]: pos=(\(String(format: "%.3f", position.x)), \(String(format: "%.3f", position.y)), \(String(format: "%.3f", position.z)))")
                    let camPos = frame.camera.transform.columns.3
                    print("   Camera at capture: (\(String(format: "%.3f", camPos.x)), \(String(format: "%.3f", camPos.y)), \(String(format: "%.3f", camPos.z)))")
                    let dist = simd_distance(SIMD3<Float>(camPos.x, camPos.y, camPos.z), position)
                    print("   Distance: \(String(format: "%.3f", dist))m")
                }

                // Collect for Gaussian Splatting
                newPoints.append(position)
                newColors.append(color)
            }
            
            // Add to Gaussian Splatting if active
            if self.isGaussianSplattingEnabled, let splatting = self.gaussianSplatRenderer, !newPoints.isEmpty {
                let beforeCount = splatting.splatCount
                splatting.addPoints(positions: newPoints, colors: newColors)
                let afterCount = splatting.splatCount
                splatUpdateFrameCount += 1

                print("✨ Splat Update:")
                print("   • Added: \(newPoints.count) new splats")
                print("   • Total: \(beforeCount) → \(afterCount)")
                print("   • Update frame: #\(splatUpdateFrameCount)")
            }

            // Add to OpenSplat Gaussian buffers if using OpenSplat renderer
            print("🔍 DEBUG: useOpenSplatRenderer=\(self.useOpenSplatRenderer), newPoints.count=\(newPoints.count)")
            if self.useOpenSplatRenderer, !newPoints.isEmpty {
                let beforeCount = self.currentGaussianCount

                // Convert points to Gaussians and add to buffers
                if let addedCount = self.convertPointsToGaussians(
                    positions: newPoints,
                    colors: newColors,
                    startIndex: self.currentGaussianCount
                ) {
                    self.currentGaussianCount += addedCount

                    // Validate the newly added range
                    let validationPassed = self.validateGaussianRange(
                        startIndex: beforeCount,
                        count: addedCount
                    )

                    print("🎯 OpenSplat Update:")
                    print("   • Added: \(addedCount) Gaussians")
                    print("   • Total: \(beforeCount) → \(self.currentGaussianCount)")
                    print("   • Validation: \(validationPassed ? "✅ PASSED" : "⚠️ FAILED")")
                    print("   • Update frame: #\(splatUpdateFrameCount)")
                } else {
                    print("❌ OpenSplat Update FAILED - could not convert points to Gaussians")
                }
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

    // MARK: - OpenSplat Buffer Management (Step 1)

    /// Allocate Metal buffers for Gaussian data
    /// Returns: true if all buffers allocated successfully, false otherwise
    private func allocateGaussianBuffers() -> Bool {
        print("📦 Allocating buffers for \(maxGaussians) Gaussians...")

        // Calculate memory requirements
        let meansSize = maxGaussians * MemoryLayout<SIMD3<Float>>.stride
        let scalesSize = maxGaussians * MemoryLayout<SIMD3<Float>>.stride
        let quatsSize = maxGaussians * MemoryLayout<SIMD4<Float>>.stride
        let colorsSize = maxGaussians * MemoryLayout<SIMD3<Float>>.stride
        let opacitiesSize = maxGaussians * MemoryLayout<Float>.stride
        let totalMB = Double(meansSize + scalesSize + quatsSize + colorsSize + opacitiesSize) / (1024 * 1024)

        print("   • Memory required: \(String(format: "%.2f", totalMB)) MB")
        print("   • Buffer breakdown:")
        print("      - Means:     \(meansSize / 1024) KB")
        print("      - Scales:    \(scalesSize / 1024) KB")
        print("      - Quats:     \(quatsSize / 1024) KB")
        print("      - Colors:    \(colorsSize / 1024) KB")
        print("      - Opacities: \(opacitiesSize / 1024) KB")

        // Allocate buffers with .storageModeShared for CPU access (debugging)
        gaussianMeansBuffer = device.makeBuffer(
            length: meansSize,
            options: .storageModeShared
        )
        guard gaussianMeansBuffer != nil else {
            print("   ❌ Failed to allocate means buffer")
            return false
        }
        print("   ✓ Means buffer allocated")

        gaussianScalesBuffer = device.makeBuffer(
            length: scalesSize,
            options: .storageModeShared
        )
        guard gaussianScalesBuffer != nil else {
            print("   ❌ Failed to allocate scales buffer")
            return false
        }
        print("   ✓ Scales buffer allocated")

        gaussianQuatsBuffer = device.makeBuffer(
            length: quatsSize,
            options: .storageModeShared
        )
        guard gaussianQuatsBuffer != nil else {
            print("   ❌ Failed to allocate quats buffer")
            return false
        }
        print("   ✓ Quats buffer allocated")

        gaussianColorsBuffer = device.makeBuffer(
            length: colorsSize,
            options: .storageModeShared
        )
        guard gaussianColorsBuffer != nil else {
            print("   ❌ Failed to allocate colors buffer")
            return false
        }
        print("   ✓ Colors buffer allocated")

        gaussianOpacitiesBuffer = device.makeBuffer(
            length: opacitiesSize,
            options: .storageModeShared
        )
        guard gaussianOpacitiesBuffer != nil else {
            print("   ❌ Failed to allocate opacities buffer")
            return false
        }
        print("   ✓ Opacities buffer allocated")

        return true
    }

    /// Validate buffer integrity and properties
    private func validateGaussianBuffers() {
        print("\n🔍 VALIDATION: Checking buffer integrity...")

        var validationPassed = true

        // Check buffer existence
        guard let meansBuffer = gaussianMeansBuffer,
              let scalesBuffer = gaussianScalesBuffer,
              let quatsBuffer = gaussianQuatsBuffer,
              let colorsBuffer = gaussianColorsBuffer,
              let opacitiesBuffer = gaussianOpacitiesBuffer else {
            print("   ❌ One or more buffers are nil")
            validationPassed = false
            return
        }

        // Check buffer lengths
        let expectedMeansSize = maxGaussians * MemoryLayout<SIMD3<Float>>.stride
        let expectedScalesSize = maxGaussians * MemoryLayout<SIMD3<Float>>.stride
        let expectedQuatsSize = maxGaussians * MemoryLayout<SIMD4<Float>>.stride
        let expectedColorsSize = maxGaussians * MemoryLayout<SIMD3<Float>>.stride
        let expectedOpacitiesSize = maxGaussians * MemoryLayout<Float>.stride

        if meansBuffer.length != expectedMeansSize {
            print("   ❌ Means buffer size mismatch: \(meansBuffer.length) vs \(expectedMeansSize)")
            validationPassed = false
        }
        if scalesBuffer.length != expectedScalesSize {
            print("   ❌ Scales buffer size mismatch")
            validationPassed = false
        }
        if quatsBuffer.length != expectedQuatsSize {
            print("   ❌ Quats buffer size mismatch")
            validationPassed = false
        }
        if colorsBuffer.length != expectedColorsSize {
            print("   ❌ Colors buffer size mismatch")
            validationPassed = false
        }
        if opacitiesBuffer.length != expectedOpacitiesSize {
            print("   ❌ Opacities buffer size mismatch")
            validationPassed = false
        }

        // Test write/read access (write sentinel values)
        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)
        meansPtr[0] = SIMD3<Float>(1.0, 2.0, 3.0)
        if meansPtr[0] == SIMD3<Float>(1.0, 2.0, 3.0) {
            print("   ✓ Means buffer read/write verified")
        } else {
            print("   ❌ Means buffer read/write FAILED")
            validationPassed = false
        }

        let scalesPtr = scalesBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: 1)
        scalesPtr[0] = SIMD3<Float>(0.5, 0.5, 0.5)
        if scalesPtr[0] == SIMD3<Float>(0.5, 0.5, 0.5) {
            print("   ✓ Scales buffer read/write verified")
        } else {
            print("   ❌ Scales buffer read/write FAILED")
            validationPassed = false
        }

        let quatsPtr = quatsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: 1)
        quatsPtr[0] = SIMD4<Float>(0, 0, 0, 1)
        if quatsPtr[0] == SIMD4<Float>(0, 0, 0, 1) {
            print("   ✓ Quats buffer read/write verified")
        } else {
            print("   ❌ Quats buffer read/write FAILED")
            validationPassed = false
        }

        if validationPassed {
            print("   ✅ All buffers validated successfully!")
        } else {
            print("   ⚠️ Buffer validation encountered issues")
        }
    }

    // MARK: - Gaussian Conversion Helper

    /// Converts simple point cloud data (positions + colors) to full Gaussian parameters
    /// - Parameters:
    ///   - positions: Array of 3D positions
    ///   - colors: Array of RGB colors
    ///   - startIndex: Starting index in the Gaussian buffers to write to
    /// - Returns: Number of Gaussians successfully added, or nil if validation fails
    private func convertPointsToGaussians(
        positions: [SIMD3<Float>],
        colors: [SIMD3<Float>],
        startIndex: Int = 0
    ) -> Int? {

        guard positions.count == colors.count else {
            print("❌ convertPointsToGaussians: position/color count mismatch")
            return nil
        }

        guard startIndex + positions.count <= maxGaussians else {
            print("❌ convertPointsToGaussians: would exceed maxGaussians (\(maxGaussians))")
            return nil
        }

        guard let meansBuffer = gaussianMeansBuffer,
              let scalesBuffer = gaussianScalesBuffer,
              let quatsBuffer = gaussianQuatsBuffer,
              let colorsBuffer = gaussianColorsBuffer,
              let opacitiesBuffer = gaussianOpacitiesBuffer else {
            print("❌ convertPointsToGaussians: buffers not allocated")
            return nil
        }

        let count = positions.count
        print("🔄 Converting \(count) points to Gaussians starting at index \(startIndex)")

        // Get typed pointers to the buffers
        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)
        let scalesPtr = scalesBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)
        let quatsPtr = quatsBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: maxGaussians)
        let colorsPtr = colorsBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)
        let opacitiesPtr = opacitiesBuffer.contents().bindMemory(to: Float.self, capacity: maxGaussians)

        // Default Gaussian parameters for point cloud conversion
        // These values create small, spherical, fully-opaque Gaussians at each point
        let defaultScale = SIMD3<Float>(0.005, 0.005, 0.005)  // 5mm radius
        let identityQuat = SIMD4<Float>(1, 0, 0, 0)           // Identity quaternion (w, x, y, z) format
        let defaultOpacity: Float = 1.0                        // Fully opaque

        // Convert each point to a Gaussian
        for i in 0..<count {
            let idx = startIndex + i

            // means3d = position (direct copy)
            meansPtr[idx] = positions[i]

            // scales = small default (creates spherical splats)
            scalesPtr[idx] = defaultScale

            // quats = identity (no rotation)
            quatsPtr[idx] = identityQuat

            // colors = RGB (direct copy)
            colorsPtr[idx] = colors[i]

            // opacities = fully opaque
            opacitiesPtr[idx] = defaultOpacity
        }

        print("   ✅ Converted \(count) points to Gaussians")
        print("   • Means: direct position copy")
        print("   • Scales: uniform \(defaultScale) (5mm radius)")
        print("   • Quats: identity (no rotation)")
        print("   • Colors: direct RGB copy")
        print("   • Opacities: \(defaultOpacity) (fully opaque)")

        return count
    }

    /// Validation helper - checks if a specific range of Gaussians contains valid data
    private func validateGaussianRange(startIndex: Int, count: Int) -> Bool {
        guard let meansBuffer = gaussianMeansBuffer,
              let colorsBuffer = gaussianColorsBuffer else {
            return false
        }

        let meansPtr = meansBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)
        let colorsPtr = colorsBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: maxGaussians)

        var validCount = 0
        for i in 0..<min(count, 10) { // Check first 10 entries
            let idx = startIndex + i
            let pos = meansPtr[idx]
            let col = colorsPtr[idx]

            // Check if position and color are non-zero
            if pos.x != 0 || pos.y != 0 || pos.z != 0 || col.x != 0 || col.y != 0 || col.z != 0 {
                validCount += 1
            }
        }

        let validRatio = Float(validCount) / Float(min(count, 10))
        let isValid = validRatio > 0.5 // At least 50% should be non-zero

        if isValid {
            print("   ✓ Gaussian range validation passed (\(validCount)/\(min(count, 10)) non-zero)")
        } else {
            print("   ⚠️ Gaussian range validation: only \(validCount)/\(min(count, 10)) non-zero")
        }

        return isValid
    }

    // MARK: - Camera Intrinsics Helper

    /// Extracts camera intrinsics and matrices for OpenSplat rendering
    /// - Parameter frame: ARFrame containing camera information
    /// - Returns: Tuple of (fx, fy, cx, cy, viewMatrix, projMatrix, imgWidth, imgHeight)
    private func extractCameraParameters(from frame: ARFrame) -> (
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        viewMatrix: simd_float4x4,
        projMatrix: simd_float4x4,
        imgWidth: Int,
        imgHeight: Int
    ) {
        let camera = frame.camera
        let intrinsics = camera.intrinsics

        // Extract focal lengths and principal point from intrinsics matrix
        // Matrix format:
        // [fx  0  cx]
        // [ 0 fy  cy]
        // [ 0  0   1]
        let fx = intrinsics[0, 0]  // focal length X
        let fy = intrinsics[1, 1]  // focal length Y
        let cx = intrinsics[2, 0]  // principal point X
        let cy = intrinsics[2, 1]  // principal point Y

        // Get view and projection matrices with correct orientation
        // Keep ARKit matrices as-is - kernel handles -Z forward convention
        let viewMatrix = camera.viewMatrix(for: orientation)
        let projMatrix = camera.projectionMatrix(
            for: orientation,
            viewportSize: viewportSize,
            zNear: 0.001,
            zFar: 1000.0
        )
        
        // DEBUG: Compare focal length derivation methods
        // MetalSplatter derives from projection matrix: focalX = screenWidth * projMatrix[0][0] / 2
        // OpenSplat needs to match this to get correct projection
        let focalFromProj = Float(viewportSize.width) * projMatrix[0][0] / 2.0
        let focalFromProjY = Float(viewportSize.height) * projMatrix[1][1] / 2.0
        print("🔍 FOCAL LENGTH COMPARISON:")
        print("   • fx from intrinsics: \(fx)")
        print("   • fx from projMatrix: \(focalFromProj) ← Using this for OpenSplat")
        print("   • projMatrix[0][0]: \(projMatrix[0][0])")
        print("   • screenWidth: \(viewportSize.width)")
        
        // Use projection-matrix-derived focal lengths to match MetalSplatter's behavior
        let fxAdjusted = focalFromProj
        let fyAdjusted = focalFromProjY

        // Get image dimensions - DO NOT swap for orientation
        // viewportSize already represents the actual render target dimensions
        // The view/proj matrices already account for orientation
        let imgWidth = Int(viewportSize.width)
        let imgHeight = Int(viewportSize.height)
        
        print("   • orientation: \(orientation.isPortrait ? "portrait" : "landscape")")
        print("   • imgWidth: \(imgWidth), imgHeight: \(imgHeight)")
        print("   • viewportSize: \(viewportSize.width) × \(viewportSize.height)")

        return (
            fx: fxAdjusted,  // Use projection-derived focal lengths
            fy: fyAdjusted,
            cx: cx,
            cy: cy,
            viewMatrix: viewMatrix,
            projMatrix: projMatrix,
            imgWidth: imgWidth,
            imgHeight: imgHeight
        )
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
    
    func makeSamplingComputePipelineState() -> MTLComputePipelineState? {
        guard let kernelFunction = library.makeFunction(name: "bernoulliSample") else {
            print("⚠️ Failed to create bernoulliSample function")
            return nil
        }
        
        return try? device.makeComputePipelineState(function: kernelFunction)
    }

    /// Makes sample points on camera image, also precompute the anchor point for animation
    func makeGridPoints(frame: ARFrame? = nil) -> [Float2] {
        // Custom Bernoulli sampling from probability buffer
        // Use LoG sampling for both MVS and LiDAR modes if promptDA is available
        if let frame = frame, let promptDA = promptDAEngine {
            
            var targetSize = CGSize(width: 512, height: 512)
            
            do {  
                // Generate probability map on the fly - now passing CVPixelBuffer directly
                let probPB = try promptDA.makeNewLoGProbability(from: frame.capturedImage, size: targetSize)

                CVPixelBufferLockBaseAddress(probPB, .readOnly)
                defer { CVPixelBufferUnlockBaseAddress(probPB, .readOnly) }
                
                // Debug: Check if probability buffer has non-zero values
                let debugPtr = CVPixelBufferGetBaseAddress(probPB)?.assumingMemoryBound(to: Float.self)
                if let ptr = debugPtr {
                    var stats = (min: Float.infinity, max: -Float.infinity, sum: Float(0), nonZero: 0)
                    let sampleSize = min(1000, CVPixelBufferGetWidth(probPB) * CVPixelBufferGetHeight(probPB))
                    for i in 0..<sampleSize {
                        let val = ptr[i]
                        stats.min = min(stats.min, val)
                        stats.max = max(stats.max, val)
                        stats.sum += val
                        if val > 0.001 { stats.nonZero += 1 }
                    }
                    print("      → Probability stats (first \(sampleSize) pixels):")
                    print("         min=\(stats.min), max=\(stats.max), avg=\(stats.sum/Float(sampleSize)), nonZero=\(stats.nonZero)")
                }
                
                let width = CVPixelBufferGetWidth(probPB)
                let height = CVPixelBufferGetHeight(probPB)
                let stride = CVPixelBufferGetBytesPerRow(probPB) / MemoryLayout<Float>.stride
                
                guard let ptr = CVPixelBufferGetBaseAddress(probPB)?.assumingMemoryBound(to: Float.self) else {
                    return []
                }
                
                let scaleX = Float(cameraResolution.x) / Float(width)
                let scaleY = Float(cameraResolution.y) / Float(height)
                
                let samplingStartTime = CFAbsoluteTimeGetCurrent()
                
                // GPU-accelerated Bernoulli sampling
                let points = performGPUSampling(
                    probabilityData: ptr,
                    width: width,
                    height: height,
                    stride: stride,
                    scaleX: scaleX,
                    scaleY: scaleY
                )
                
                let samplingTime = CFAbsoluteTimeGetCurrent() - samplingStartTime
                print("      → GPU Bernoulli sampling time: \(String(format: "%.3f", samplingTime))s")
                
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
    
    /// Perform GPU-accelerated Bernoulli sampling using Metal compute shader
    func performGPUSampling(probabilityData: UnsafePointer<Float>,
                           width: Int,
                           height: Int,
                           stride: Int,
                           scaleX: Float,
                           scaleY: Float) -> [Float2] {
        
        let maxPossiblePoints = width * height
        let dataSize = stride * height * MemoryLayout<Float>.stride
        
        // Create or reuse buffers
        if probabilityBuffer == nil || probabilityBuffer!.length < dataSize {
            probabilityBuffer = device.makeBuffer(length: dataSize, options: .storageModeShared)
        }
        
        if sampledPointsBuffer == nil || sampledPointsBuffer!.length < maxPossiblePoints * MemoryLayout<Float2>.stride {
            sampledPointsBuffer = device.makeBuffer(length: maxPossiblePoints * MemoryLayout<Float2>.stride, 
                                                    options: .storageModeShared)
        }
        
        if atomicCounterBuffer == nil {
            atomicCounterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, 
                                                    options: .storageModeShared)
        }
        
        guard let probBuffer = probabilityBuffer,
              let pointsBuffer = sampledPointsBuffer,
              let counterBuffer = atomicCounterBuffer else {
            print("⚠️ Failed to create Metal buffers for GPU sampling")
            return []
        }
        
        // Copy probability data to GPU
        memcpy(probBuffer.contents(), probabilityData, dataSize)
        
        // Reset atomic counter to 0
        counterBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)
        
        // Setup uniforms
        var uniforms = SamplingUniforms(
            width: UInt32(width),
            height: UInt32(height),
            stride: UInt32(stride),
            scaleX: scaleX,
            scaleY: scaleY,
            maxPoints: UInt32(maxPossiblePoints)
        )
        
        // Create command buffer and compute encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("⚠️ Failed to create command buffer/encoder")
            return []
        }
        
        computeEncoder.setComputePipelineState(samplingComputePipelineState)
        computeEncoder.setBuffer(probBuffer, offset: 0, index: ComputeBufferIndices.kProbabilityMap.rawValue)
        computeEncoder.setBuffer(pointsBuffer, offset: 0, index: ComputeBufferIndices.kSampledPoints.rawValue)
        computeEncoder.setBuffer(counterBuffer, offset: 0, index: ComputeBufferIndices.kAtomicCounter.rawValue)
        computeEncoder.setBytes(&uniforms, length: MemoryLayout<SamplingUniforms>.stride, index: ComputeBufferIndices.kSamplingUniforms.rawValue)
        
        // Calculate thread groups
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupCount = MTLSize(
            width: (width + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (height + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        computeEncoder.endEncoding()
        
        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read back results
        let pointCount = counterBuffer.contents().load(as: UInt32.self)
        let pointsPointer = pointsBuffer.contents().assumingMemoryBound(to: Float2.self)
        let points = Array(UnsafeBufferPointer(start: pointsPointer, count: Int(pointCount)))
        
        print("📊 GPU sampled \(points.count) points")
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
