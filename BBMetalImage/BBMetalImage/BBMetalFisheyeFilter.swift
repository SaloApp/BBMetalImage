//
//  BBMetalFisheyeFilter.swift
//  BBMetalImage
//
//  Created based on fisheye effect implementation
//

import Metal

/// Applies a fisheye distortion effect on an image
public class BBMetalFisheyeFilter: BBMetalBaseFilter {
    private struct FisheyeUniforms {
        var modifier: Float
        var borderSoftness: Float
        var vignetteStrength: Float
    }

    /// Fisheye effect modifier (0.0 ~ 1.0, default 0.5)
    /// Higher values create stronger fisheye distortion
    public var modifier: Float

    /// Softness of radial border fade (0.0 ~ 1.0).
    public var borderSoftness: Float

    /// Strength of edge darkening inside the lens region (0.0 ~ 1.0).
    public var vignetteStrength: Float

    public init(
        modifier: Float = 0.5,
        borderSoftness: Float = 0.18,
        vignetteStrength: Float = 0.22
    ) {
        self.modifier = modifier
        self.borderSoftness = borderSoftness
        self.vignetteStrength = vignetteStrength
        super.init(kernelFunctionName: "fisheyeKernel")
    }

    public override func updateParameters(for encoder: MTLComputeCommandEncoder, texture: BBMetalTexture) {
        var uniforms = FisheyeUniforms(
            modifier: modifier.clamped(to: 0 ... 1),
            borderSoftness: borderSoftness.clamped(to: 0.02 ... 0.9),
            vignetteStrength: vignetteStrength.clamped(to: 0 ... 1)
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<FisheyeUniforms>.stride, index: 0)
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
