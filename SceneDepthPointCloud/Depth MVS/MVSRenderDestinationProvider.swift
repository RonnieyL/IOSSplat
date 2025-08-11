//
//  MVSRenderDestinationProvider.swift
//  SceneDepthPointCloud
//
//  Created by Ronak Sinha on 8/10/25.
//  Copyright © 2025 Apple. All rights reserved.
//

import MetalKit

protocol MVSRenderDestinationProvider: AnyObject {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get }
    var depthStencilPixelFormat: MTLPixelFormat { get }
    var sampleCount: Int { get }
}

// Make MTKView the render destination.
extension MTKView: MVSRenderDestinationProvider {}
