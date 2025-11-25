//
//  ExportLogic.swift
//  SceneDepthPointCloud
//
//  Created by GitHub Copilot on 11/24/25.
//

import Foundation
import ARKit
import Metal
import MetalKit
import Photos
import UIKit

extension Renderer {
    
    // MARK: - Depth Photo Saving
    
    // Save ARKit depth map as a photo in Photos app
    func saveDepthAsPhoto(_ depthPixelBuffer: CVPixelBuffer) {
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
    
    // Request to save the next depth frame as a photo
    func saveDepthPhoto() {
        shouldSaveNextDepth = true
        print("📸 Depth photo will be saved on next frame")
    }
    
    // MARK: - PLY Saving
    
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

    func extractCameraData(frame: ARFrame) {
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
