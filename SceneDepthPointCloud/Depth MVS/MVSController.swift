import UIKit
import Metal
import MetalKit
import ARKit

final class MVSController: UIViewController, ARSessionDelegate {
    private let isUIEnabled = true
    private var clearButton = UIButton(type: .system)
    private let confidenceControl = UISegmentedControl(items: ["Low", "Medium", "High"])
    private let fpsControl = UISegmentedControl(items: ["1 FPS", "3 FPS", "5 FPS", "10 FPS", "30 FPS"])
    private var rgbButton = UIButton(type: .system)
    private var showSceneButton = UIButton(type: .system)
    private var saveButton = UIButton(type: .system)
    private var toggleParticlesButton = UIButton(type: .system)
    private let session = ARSession()
    var renderer: MVSRenderer!
    private var isPaused = false
    
    override func viewDidLoad() {
        super.viewDidLoad()
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        session.delegate = self
        // Set the view to use the default device
        if let view = view as? MTKView {
            view.device = device
            view.backgroundColor = UIColor.clear
            // we need this to enable depth test
            view.depthStencilPixelFormat = .depth32Float
            view.contentScaleFactor = 1
            view.delegate = self
            // Configure the renderer to draw to the view
            renderer = MVSRenderer(session: session, metalDevice: device, renderDestination: view)
            renderer.drawRectResized(size: view.bounds.size)
            
            // Set initial MVS processing rate to 3 FPS
            renderer.mvsProcessingFPS = 3.0
        }
        
        clearButton = createButton(mainView: self, iconName: "trash.circle.fill",
            tintColor: .red, hidden: !isUIEnabled)
        view.addSubview(clearButton)
        
        saveButton = createButton(mainView: self, iconName: "tray.and.arrow.down.fill",
            tintColor: .white, hidden: !isUIEnabled)
        view.addSubview(saveButton)
        
        toggleParticlesButton = createButton(mainView: self, iconName: "circle.grid.hex.fill",
            tintColor: .systemBlue, hidden: !isUIEnabled)
        view.addSubview(toggleParticlesButton)
        
        showSceneButton = createButton(mainView: self, iconName: "play.fill",
            tintColor: .green, hidden: !isUIEnabled)
        view.addSubview(showSceneButton)
        
        rgbButton = createButton(mainView: self, iconName: "camera.fill",
            tintColor: .white, hidden: !isUIEnabled)
        view.addSubview(rgbButton)
        
        // Set up confidence control
        confidenceControl.selectedSegmentIndex = 2
        confidenceControl.isHidden = !isUIEnabled
        confidenceControl.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        confidenceControl.selectedSegmentTintColor = UIColor.white
        confidenceControl.setTitleTextAttributes([.foregroundColor: UIColor.black], for: .selected)
        confidenceControl.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        view.addSubview(confidenceControl)
        
        // Set up FPS control
        fpsControl.selectedSegmentIndex = 1 // Default to 3 FPS
        fpsControl.isHidden = !isUIEnabled
        fpsControl.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        fpsControl.selectedSegmentTintColor = UIColor.white
        fpsControl.setTitleTextAttributes([.foregroundColor: UIColor.black], for: .selected)
        fpsControl.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        view.addSubview(fpsControl)
        
        setupButtons()
        setupLayout()
    }
    
    func setupButtons() {
        clearButton.addTarget(self, action: #selector(onClearButtonPressed), for: .touchUpInside)
        showSceneButton.addTarget(self, action: #selector(onShowSceneButtonPressed), for: .touchUpInside)
        saveButton.addTarget(self, action: #selector(onSaveButtonPressed), for: .touchUpInside)
        toggleParticlesButton.addTarget(self, action: #selector(onToggleParticlesButtonPressed), for: .touchUpInside)
        rgbButton.addTarget(self, action: #selector(onRgbButtonPressed), for: .touchUpInside)
        confidenceControl.addTarget(self, action: #selector(onConfidenceChanged), for: .valueChanged)
        fpsControl.addTarget(self, action: #selector(onFPSChanged), for: .valueChanged)
    }
    
    func setupLayout() {
        clearButton.translatesAutoresizingMaskIntoConstraints = false
        showSceneButton.translatesAutoresizingMaskIntoConstraints = false
        saveButton.translatesAutoresizingMaskIntoConstraints = false
        toggleParticlesButton.translatesAutoresizingMaskIntoConstraints = false
        rgbButton.translatesAutoresizingMaskIntoConstraints = false
        confidenceControl.translatesAutoresizingMaskIntoConstraints = false
        fpsControl.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            clearButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            clearButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            clearButton.widthAnchor.constraint(equalToConstant: 50),
            clearButton.heightAnchor.constraint(equalToConstant: 50),
            
            saveButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            saveButton.topAnchor.constraint(equalTo: clearButton.bottomAnchor, constant: 20),
            saveButton.widthAnchor.constraint(equalToConstant: 50),
            saveButton.heightAnchor.constraint(equalToConstant: 50),
            
            toggleParticlesButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            toggleParticlesButton.topAnchor.constraint(equalTo: saveButton.bottomAnchor, constant: 20),
            toggleParticlesButton.widthAnchor.constraint(equalToConstant: 50),
            toggleParticlesButton.heightAnchor.constraint(equalToConstant: 50),
            
            showSceneButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            showSceneButton.topAnchor.constraint(equalTo: toggleParticlesButton.bottomAnchor, constant: 20),
            showSceneButton.widthAnchor.constraint(equalToConstant: 50),
            showSceneButton.heightAnchor.constraint(equalToConstant: 50),
            
            rgbButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            rgbButton.topAnchor.constraint(equalTo: showSceneButton.bottomAnchor, constant: 20),
            rgbButton.widthAnchor.constraint(equalToConstant: 50),
            rgbButton.heightAnchor.constraint(equalToConstant: 50),
            
            confidenceControl.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            confidenceControl.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -90),
            confidenceControl.widthAnchor.constraint(equalToConstant: 280),
            confidenceControl.heightAnchor.constraint(equalToConstant: 32),
            
            fpsControl.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            fpsControl.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -50),
            fpsControl.widthAnchor.constraint(equalToConstant: 280),
            fpsControl.heightAnchor.constraint(equalToConstant: 32),
        ])
    }
    
    @objc
    private func onClearButtonPressed() {
        renderer.clearParticles()
        renderer.stopDataExtraction()
    }
    
    @objc
    private func onShowSceneButtonPressed() {
        renderer.isInViewSceneMode = !renderer.isInViewSceneMode
        if !renderer.isInViewSceneMode {
            renderer.showParticles = true
            self.toggleParticlesButton.setBackgroundImage(.init(systemName: "circle.grid.hex.fill"), for: .normal)
            self.setShowSceneButtonStyle(isScanning: true)
            // Start data extraction when scanning starts
            renderer.dataExtractionFPS = renderer.mvsProcessingFPS // Use same FPS as MVS
            renderer.startDataExtraction()
        } else {
            self.setShowSceneButtonStyle(isScanning: false)
            // Stop data extraction when scanning stops
            renderer.stopDataExtraction()
        }
    }
    
    @objc
    private func onSaveButtonPressed() {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let saveController = storyboard.instantiateViewController(withIdentifier: "SaveController") as! SaveController
        saveController.renderer = self.renderer
        self.present(saveController, animated: true, completion: nil)
    }
    
    @objc
    private func onToggleParticlesButtonPressed() {
        renderer.showParticles = !renderer.showParticles
        if renderer.showParticles {
            self.toggleParticlesButton.setBackgroundImage(.init(systemName: "circle.grid.hex.fill"), for: .normal)
        } else {
            self.toggleParticlesButton.setBackgroundImage(.init(systemName: "circle.grid.hex"), for: .normal)
            renderer.stopDataExtraction()
        }
    }
    
    @objc
    private func onRgbButtonPressed() {
        renderer.rgbUniforms.radius = renderer.rgbUniforms.radius <= 0 ? 1 : 0
        if renderer.rgbUniforms.radius > 0 {
            rgbButton.setBackgroundImage(.init(systemName: "camera.fill"), for: .normal)
        } else {
            rgbButton.setBackgroundImage(.init(systemName: "camera"), for: .normal)
        }
    }
    
    @objc
    private func onConfidenceChanged() {
        renderer.confidenceThreshold = confidenceControl.selectedSegmentIndex
    }
    
    @objc
    private func onFPSChanged() {
        let fpsValues: [Double] = [1.0, 3.0, 5.0, 10.0, 30.0]
        renderer.mvsProcessingFPS = fpsValues[fpsControl.selectedSegmentIndex]
    }
    
    private func setShowSceneButtonStyle(isScanning: Bool) {
        if isScanning {
            showSceneButton.setBackgroundImage(.init(systemName: "stop.fill"), for: .normal)
            showSceneButton.tintColor = .red
        } else {
            showSceneButton.setBackgroundImage(.init(systemName: "play.fill"), for: .normal)
            showSceneButton.tintColor = .green
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Note: MVS requires camera permission but doesn't need LiDAR
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = []  // No LiDAR depth for MVS mode
        
        session.run(configuration)
        
        guard ARWorldTrackingConfiguration.isSupported else {
            fatalError("ARKit is not available on this device.")
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        session.pause()
    }
    
    // MARK: - ARSessionDelegate
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        guard error is ARError else { return }
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        DispatchQueue.main.async {
            print("ARSession failed with error: \(errorMessage)")
        }
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        isPaused = true
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        isPaused = false
    }
}

// MARK: - MTKViewDelegate
extension MVSController: MTKViewDelegate {
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }
    
    func draw(in view: MTKView) {
        if !isPaused {
            renderer.draw()
        }
    }
}

// MARK: - RenderDestinationProvider
extension MVSController: RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? {
        return view.currentRenderPassDescriptor
    }
    
    var currentDrawable: CAMetalDrawable? {
        return view.currentDrawable
    }
    
    var colorPixelFormat: MTLPixelFormat {
        return view.colorPixelFormat
    }
    
    var depthStencilPixelFormat: MTLPixelFormat {
        return view.depthStencilPixelFormat
    }
    
    var sampleCount: Int {
        return view.sampleCount
    }
}

func createButton(mainView: UIViewController, iconName: String, tintColor: UIColor, hidden: Bool) -> UIButton {
    let button = UIButton(type: .system)
    button.setBackgroundImage(.init(systemName: iconName), for: .normal)
    button.tintColor = tintColor
    button.contentHorizontalAlignment = .fill
    button.contentVerticalAlignment = .fill
    button.isHidden = hidden
    return button
}
