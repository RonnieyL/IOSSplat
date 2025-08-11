import UIKit
import Metal
import MetalKit
import ARKit

final class MVSController: UIViewController, ARSessionDelegate {

    // MARK: - IBOutlets
    // Connect this to a MetalKit View in your storyboard (Class = MTKView)
    @IBOutlet weak var mtkView: MTKView!

    // MARK: - UI
    private let isUIEnabled = true
    private var clearButton = UIButton(type: .system)
    private let confidenceControl = UISegmentedControl(items: ["Low", "Medium", "High"])
    private let fpsControl = UISegmentedControl(items: ["1 FPS", "3 FPS", "5 FPS", "10 FPS", "30 FPS"])
    private var rgbButton = UIButton(type: .system)
    private var showSceneButton = UIButton(type: .system)
    private var saveButton = UIButton(type: .system)
    private var toggleParticlesButton = UIButton(type: .system)

    // MARK: - AR / Rendering
    private let session = ARSession()
    var renderer: MVSRenderer!
    private var isPaused = false

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()

        // Create the MTKView in code if the outlet isn't wired
        if mtkView == nil {
            let v = MTKView(frame: view.bounds)
            v.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview(v)
            NSLayoutConstraint.activate([
                v.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                v.trailingAnchor.constraint(equalTo: view.trailingAnchor),
                v.topAnchor.constraint(equalTo: view.topAnchor),
                v.bottomAnchor.constraint(equalTo: view.bottomAnchor)
            ])
            mtkView = v
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        session.delegate = self

        // MTKView setup
        mtkView.device = device
        mtkView.backgroundColor = .clear
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.contentScaleFactor = 1
        mtkView.sampleCount = 1
        mtkView.delegate = self

        // Renderer draws into the MTKView (which conforms to MVSRenderDestinationProvider)
        renderer = MVSRenderer(session: session, metalDevice: device, renderDestination: mtkView)
        renderer.drawRectResized(size: mtkView.drawableSize)
        renderer.mvsProcessingFPS = 3.0

        // UI
        clearButton = createButton(mainView: self, iconName: "trash.circle.fill",
                                   tintColor: .red, hidden: !isUIEnabled)
        saveButton = createButton(mainView: self, iconName: "tray.and.arrow.down.fill",
                                  tintColor: .white, hidden: !isUIEnabled)
        toggleParticlesButton = createButton(mainView: self, iconName: "circle.grid.hex.fill",
                                             tintColor: .systemBlue, hidden: !isUIEnabled)
        showSceneButton = createButton(mainView: self, iconName: "play.fill",
                                       tintColor: .green, hidden: !isUIEnabled)
        rgbButton = createButton(mainView: self, iconName: "camera.fill",
                                 tintColor: .white, hidden: !isUIEnabled)

        [clearButton, saveButton, toggleParticlesButton, showSceneButton, rgbButton].forEach {
            view.addSubview($0)
        }

        // Confidence control
        confidenceControl.selectedSegmentIndex = 2
        confidenceControl.isHidden = !isUIEnabled
        confidenceControl.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        confidenceControl.selectedSegmentTintColor = .white
        confidenceControl.setTitleTextAttributes([.foregroundColor: UIColor.black], for: .selected)
        confidenceControl.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        view.addSubview(confidenceControl)

        // FPS control
        fpsControl.selectedSegmentIndex = 1 // 3 FPS
        fpsControl.isHidden = !isUIEnabled
        fpsControl.backgroundColor = UIColor.black.withAlphaComponent(0.5)
        fpsControl.selectedSegmentTintColor = .white
        fpsControl.setTitleTextAttributes([.foregroundColor: UIColor.black], for: .selected)
        fpsControl.setTitleTextAttributes([.foregroundColor: UIColor.white], for: .normal)
        view.addSubview(fpsControl)

        setupButtons()
        setupLayout()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        guard ARWorldTrackingConfiguration.isSupported else {
            fatalError("ARKit is not available on this device.")
        }

        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = [] // MVS doesn't need LiDAR
        session.run(configuration)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        session.pause()
    }

    // MARK: - UI wiring
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
        [clearButton, showSceneButton, saveButton, toggleParticlesButton, rgbButton,
         confidenceControl, fpsControl].forEach { $0.translatesAutoresizingMaskIntoConstraints = false }

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

    // MARK: - Actions
    @objc private func onClearButtonPressed() {
        renderer.clearParticles()
        renderer.stopDataExtraction()
    }

    @objc private func onShowSceneButtonPressed() {
        renderer.isInViewSceneMode.toggle()
        if !renderer.isInViewSceneMode {
            renderer.showParticles = true
            toggleParticlesButton.setBackgroundImage(.init(systemName: "circle.grid.hex.fill"), for: .normal)
            setShowSceneButtonStyle(isScanning: true)
            renderer.dataExtractionFPS = renderer.mvsProcessingFPS
            renderer.startDataExtraction()
        } else {
            setShowSceneButtonStyle(isScanning: false)
            renderer.stopDataExtraction()
        }
    }

    @objc private func onSaveButtonPressed() {
        let storyboard = UIStoryboard(name: "Main", bundle: nil)
        let saveController = storyboard.instantiateViewController(withIdentifier: "SaveController") as! SaveController
        // NOTE: SaveController must have:  `weak var mvsRenderer: MVSRenderer?`
        // Then use the next line:
        // saveController.mvsRenderer = self.renderer
        // If your SaveController still has `renderer: Renderer?` from LiDAR, update it accordingly.
        present(saveController, animated: true)
    }

    @objc private func onToggleParticlesButtonPressed() {
        renderer.showParticles.toggle()
        let name = renderer.showParticles ? "circle.grid.hex.fill" : "circle.grid.hex"
        toggleParticlesButton.setBackgroundImage(.init(systemName: name), for: .normal)
        if !renderer.showParticles { renderer.stopDataExtraction() }
    }

    @objc private func onRgbButtonPressed() {
        // rgbUniforms must be internal (not private) in MVSRenderer
        renderer.rgbUniforms.radius = renderer.rgbUniforms.radius <= 0 ? 1 : 0
        rgbButton.setBackgroundImage(.init(systemName: renderer.rgbUniforms.radius > 0 ? "camera.fill" : "camera"), for: .normal)
    }

    @objc private func onConfidenceChanged() {
        renderer.confidenceThreshold = confidenceControl.selectedSegmentIndex
    }

    @objc private func onFPSChanged() {
        let fpsValues: [Double] = [1.0, 3.0, 5.0, 10.0, 30.0]
        renderer.mvsProcessingFPS = fpsValues[fpsControl.selectedSegmentIndex]
    }

    private func setShowSceneButtonStyle(isScanning: Bool) {
        showSceneButton.setBackgroundImage(.init(systemName: isScanning ? "stop.fill" : "play.fill"), for: .normal)
        showSceneButton.tintColor = isScanning ? .red : .green
    }
}

// MARK: - MTKViewDelegate
extension MVSController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }
    func draw(in view: MTKView) {
        if !isPaused { renderer.draw() }
    }
}

// MARK: - Button factory
func createButton(mainView: UIViewController, iconName: String, tintColor: UIColor, hidden: Bool) -> UIButton {
    let button = UIButton(type: .system)
    button.setBackgroundImage(.init(systemName: iconName), for: .normal)
    button.tintColor = tintColor
    button.contentHorizontalAlignment = .fill
    button.contentVerticalAlignment = .fill
    button.isHidden = hidden
    return button
}
