//
// Copyright (C) 2024, Inria
// GRAPHDECO research group, https://team.inria.fr/graphdeco
// All rights reserved.
//
// This software is free for non-commercial, research and evaluation use 
// under the terms of the LICENSE.md file.
//
// For inquiries contact george.drettakis@inria.fr
//

import UIKit

class OptionSelectionController: UIViewController {
    
    @IBOutlet weak var titleLabel: UILabel!
    @IBOutlet weak var lidarButton: UIButton!
    @IBOutlet weak var depthMVSButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }
    
    private func setupUI() {
        // Set up title
        titleLabel.text = "Pick Your Option"
        titleLabel.font = UIFont.systemFont(ofSize: 32, weight: .bold)
        titleLabel.textAlignment = .center
        titleLabel.textColor = .label
        
        // Set up LiDAR button
        setupButton(lidarButton, title: "LiDAR Scanner", isEnabled: true)
        
        // Set up Depth MVS button  
        setupButton(depthMVSButton, title: "Depth MVS", isEnabled: false)
    }
    
    private func setupButton(_ button: UIButton, title: String, isEnabled: Bool) {
        button.setTitle(title, for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 20, weight: .semibold)
        button.backgroundColor = isEnabled ? UIColor.systemBlue : UIColor.systemGray3
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 12
        button.layer.shadowColor = UIColor.black.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 2)
        button.layer.shadowRadius = 4
        button.layer.shadowOpacity = 0.1
        button.isEnabled = isEnabled
        
        if !isEnabled {
            button.setTitleColor(.systemGray, for: .normal)
        }
    }
    
    @IBAction func lidarButtonTapped(_ sender: UIButton) {
        // Navigate to the existing LiDAR scanner
        performSegue(withIdentifier: "showLiDARScanner", sender: self)
    }
    
    @IBAction func depthMVSButtonTapped(_ sender: UIButton) {
        // For now, show an alert that this feature is coming soon
        let alert = UIAlertController(
            title: "Coming Soon",
            message: "Depth MVS functionality is under development.",
            preferredStyle: .alert
        )
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}
