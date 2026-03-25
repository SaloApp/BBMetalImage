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
        var distortionMix: Float
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

    /// Extra blend applied to distortion on front camera (0.0 ~ 1.0).
    /// 0 = keep image almost unchanged, 1 = full fisheye distortion.
    public var frontCameraDistortionMix: Float

    public init(
        modifier: Float = 0.5,
        borderSoftness: Float = 0.18,
        vignetteStrength: Float = 0.12,
        frontCameraDistortionMix: Float = 0.2
    ) {
        self.modifier = modifier
        self.borderSoftness = borderSoftness
        self.vignetteStrength = vignetteStrength
        self.frontCameraDistortionMix = frontCameraDistortionMix
        super.init(kernelFunctionName: "fisheyeKernel")
    }

    public override func updateParameters(for encoder: MTLComputeCommandEncoder, texture: BBMetalTexture) {
        let isFrontCamera = texture.cameraPosition == .front
        var uniforms = FisheyeUniforms(
            modifier: modifier.clamped(to: 0 ... 1),
            distortionMix: (isFrontCamera ? frontCameraDistortionMix : 1).clamped(to: 0 ... 1),
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
