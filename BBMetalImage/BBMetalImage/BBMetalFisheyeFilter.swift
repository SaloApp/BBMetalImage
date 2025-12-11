//
//  BBMetalFisheyeFilter.swift
//  BBMetalImage
//
//  Created based on fisheye effect implementation
//

import Metal

/// Applies a fisheye distortion effect on an image
public class BBMetalFisheyeFilter: BBMetalBaseFilter {
    /// Fisheye effect modifier (0.0 ~ 1.0, default 0.5)
    /// Higher values create stronger fisheye distortion
    public var modifier: Float
    
    public init(modifier: Float = 0.5) {
        self.modifier = modifier
        super.init(kernelFunctionName: "fisheyeKernel")
    }
    
    public override func updateParameters(for encoder: MTLComputeCommandEncoder, texture: BBMetalTexture) {
        var mod = modifier
        encoder.setBytes(&mod, length: MemoryLayout<Float>.size, index: 0)
    }
}
