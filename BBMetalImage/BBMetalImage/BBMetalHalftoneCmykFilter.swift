//
//  BBMetalHalftoneCmykFilter.swift
//  BBMetalImage
//
//  CMYK halftone filter ported from paper-design shaders.
//

import Metal
import simd

public final class BBMetalHalftoneCmykFilter: BBMetalBaseFilter {
	public var size: Float {
		get { _size }
		set { _size = Self.clamp(newValue, min: 0.0, max: 1.0) }
	}
	public var softness: Float {
		get { _softness }
		set { _softness = Self.clamp(newValue, min: 0.0, max: 1.0) }
	}
	public var grainSize: Float {
		get { _grainSize }
		set { _grainSize = Self.clamp(newValue, min: 0.0, max: 1.0) }
	}

	private var _size: Float
	private var _softness: Float
	private var _grainSize: Float

public init(size: Float = 0.2, softness: Float = 1.0, grainSize: Float = 0.5) {
		_size = Self.clamp(size, min: 0.0, max: 1.0)
		_softness = Self.clamp(softness, min: 0.0, max: 1.0)
		_grainSize = Self.clamp(grainSize, min: 0.0, max: 1.0)
		super.init(kernelFunctionName: "halftoneCmykKernel")
	}

	private static func clamp(_ value: Float, min: Float, max: Float) -> Float {
		Swift.max(min, Swift.min(max, value))
	}

	private struct HalftoneParams {
		var size: Float
		var softness: Float
		var grainSize: Float
		var contrast: Float
		var grainMixer: Float
		var grainOverlay: Float
		var gridNoise: Float
		var floodC: Float
		var floodM: Float
		var floodY: Float
		var floodK: Float
		var gainC: Float
		var gainM: Float
		var gainY: Float
		var gainK: Float
		var type: Float
		var mixOriginal: Float
		var colorBack: SIMD4<Float>
		var colorC: SIMD4<Float>
		var colorM: SIMD4<Float>
		var colorY: SIMD4<Float>
		var colorK: SIMD4<Float>
	}

	override public func updateParameters(for encoder: MTLComputeCommandEncoder, texture: BBMetalTexture) {
		var params = HalftoneParams(
			size: _size,
			softness: _softness,
			grainSize: _grainSize,
				contrast: 1.0,
				grainMixer: 0.0,
				grainOverlay: 0.0,
				gridNoise: 0.2,
				floodC: 0.15,
			floodM: 0.0,
			floodY: 0.0,
			floodK: 0.0,
				gainC: 0.30,
			gainM: 0.0,
				gainY: 0.20,
			gainK: 0.0,
			type: 1.0,
				mixOriginal: 0.0,
			colorBack: SIMD4<Float>(0.9843137, 0.98039216, 0.95686275, 1.0), // #fbfaf4
			colorC: SIMD4<Float>(0.0, 0.69803923, 1.0, 1.0), // #00b2ff
			colorM: SIMD4<Float>(0.9882353, 0.30980393, 0.6156863, 1.0), // #fc4f9d
			colorY: SIMD4<Float>(1.0, 0.8509804, 0.0, 1.0), // #ffd900
				colorK: SIMD4<Float>(0.13725491, 0.12156863, 0.1254902, 1.0) // #231f20
			)
		encoder.setBytes(&params, length: MemoryLayout<HalftoneParams>.stride, index: 0)
	}
}
