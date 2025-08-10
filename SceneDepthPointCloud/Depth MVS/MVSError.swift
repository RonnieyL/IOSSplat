//
//  MVSError.swift
//  SceneDepthPointCloud - Depth MVS
//

import Foundation

enum MVSError : Error {
    case savingFailed
    case noScanDone
    case alreadySavingFile
    case depthProcessingFailed
    case modelLoadingFailed
}
