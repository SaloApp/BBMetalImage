//
//  BBMetalVHSFilter.swift
//  BBMetalImage
//
//  Created based on VHS shader from shadertoy.com/view/XlsczN
//

import Metal
import QuartzCore

/// A Metal-based VHS distortion filter
/// Based on shader code from: https://www.shadertoy.com/view/XlsczN
/// This filter creates a VHS tape effect with chromatic aberration, scanlines, and distortion
public class BBMetalVHSFilter: BBMetalBaseFilter {
	/// Strength of the VHS effect (0.0 = no effect, 1.0 = maximum effect)
	/// Controls the intensity of the distortion and chromatic aberration
	public var strength: Float {
		get { _strength }
		set {
			_strength = BBMetalVHSFilter.clamp(newValue, min: 0.0, max: 1.0)
		}
	}

	/// Time parameter for animated effects (0.0+)
	/// Used for scanline movement and random distortion
	/// Automatically updates each frame
	public var time: Float {
		get { _time }
		set {
			_time = max(0.0, newValue)
		}
	}

	private var _strength: Float
	private var _time: Float
	private var startTime: CFTimeInterval?

	public init(strength: Float = 0.5, time: Float = 0.0) {
		_strength = BBMetalVHSFilter.clamp(strength, min: 0.0, max: 1.0)
		_time = max(0.0, time)
		super.init(kernelFunctionName: "vhsKernel")
	}

	private static func clamp(_ value: Float, min: Float, max: Float) -> Float {
		return Swift.max(min, Swift.min(max, value))
	}

	override public func updateParameters(for encoder: MTLComputeCommandEncoder, texture: BBMetalTexture) {
		// Update time automatically based on elapsed time since first frame
		if startTime == nil {
			startTime = CACurrentMediaTime()
		}
		let elapsed = Float(CACurrentMediaTime() - (startTime ?? 0))
		
		var str = _strength
		var t = elapsed
		encoder.setBytes(&str, length: MemoryLayout<Float>.size, index: 0)
		encoder.setBytes(&t, length: MemoryLayout<Float>.size, index: 1)
	}
}
