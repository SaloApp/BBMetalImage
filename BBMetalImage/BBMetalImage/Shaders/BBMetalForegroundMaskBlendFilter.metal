#include <metal_stdlib>
using namespace metal;

kernel void foregroundMaskBlendKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::read> inputTexture [[texture(1)]],
	texture2d<float, access::sample> inputTexture2 [[texture(2)]],
	texture2d<half, access::sample> inputTexture3 [[texture(3)]],
	constant bool *shouldFlipMaskVertically [[buffer(0)]],
	constant bool *isBackCamera [[buffer(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	const half4 base = inputTexture.read(gid);
	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
	const float width = float(outputTexture.get_width());
	const float height = float(outputTexture.get_height());
	const float2 uv = float2(
		(float(gid.x) + 0.5) / width,
		(float(gid.y) + 0.5) / height
	);
	const float2 maskUV = float2(
		(float(gid.x) + 0.5) / width,
		bool(*shouldFlipMaskVertically) ? (1.0 - uv.y) : uv.y
	);
	const float2 maskTexel = float2(
		1.0 / max(float(inputTexture2.get_width()), 1.0),
		1.0 / max(float(inputTexture2.get_height()), 1.0)
	);
	const float centerMask = clamp(inputTexture2.sample(quadSampler, maskUV).r, 0.0, 1.0);
	const float leftMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(-maskTexel.x, 0.0)).r, 0.0, 1.0);
	const float rightMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(maskTexel.x, 0.0)).r, 0.0, 1.0);
	const float topMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(0.0, -maskTexel.y)).r, 0.0, 1.0);
	const float bottomMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(0.0, maskTexel.y)).r, 0.0, 1.0);
	const float smoothedMask = (centerMask * 4.0 + leftMask + rightMask + topMask + bottomMask) / 8.0;
	const float maskThresholdLow = bool(*isBackCamera) ? 0.62 : 0.5;
	const float maskThresholdHigh = bool(*isBackCamera) ? 0.94 : 0.9;
	const float maskGamma = bool(*isBackCamera) ? 2.0 : 1.7;
	const float personMask = smoothstep(maskThresholdLow, maskThresholdHigh, pow(smoothedMask, maskGamma));
	const float outAspect = width / height;
	const float bgWidth = float(inputTexture3.get_width());
	const float bgHeight = float(inputTexture3.get_height());
	const float bgAspect = bgWidth / bgHeight;
	float2 bgUV = uv;
	if (bgAspect > outAspect) {
		const float normalizedWidth = outAspect / bgAspect;
		bgUV.x = (uv.x - 0.5) * normalizedWidth + 0.5;
	} else if (bgAspect < outAspect) {
		const float normalizedHeight = bgAspect / outAspect;
		bgUV.y = (uv.y - 0.5) * normalizedHeight + 0.5;
	}
	bgUV = clamp(bgUV, float2(0.0), float2(1.0));
	const half3 background = inputTexture3.sample(quadSampler, bgUV).rgb;
	const half3 rgb = mix(background, base.rgb, half(personMask));
	outputTexture.write(half4(rgb, base.a), gid);
}

kernel void foregroundColorPopKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::read> inputTexture [[texture(1)]],
	texture2d<float, access::sample> inputTexture2 [[texture(2)]],
	constant bool *shouldFlipMaskVertically [[buffer(0)]],
	constant bool *isBackCamera [[buffer(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	const half4 base = inputTexture.read(gid);
	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
	const float width = float(outputTexture.get_width());
	const float height = float(outputTexture.get_height());
	const float2 uv = float2(
		(float(gid.x) + 0.5) / width,
		(float(gid.y) + 0.5) / height
	);
	const float2 maskUV = float2(
		(float(gid.x) + 0.5) / width,
		bool(*shouldFlipMaskVertically) ? (1.0 - uv.y) : uv.y
	);
	const float2 maskTexel = float2(
		1.0 / max(float(inputTexture2.get_width()), 1.0),
		1.0 / max(float(inputTexture2.get_height()), 1.0)
	);
	const float centerMask = clamp(inputTexture2.sample(quadSampler, maskUV).r, 0.0, 1.0);
	const float leftMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(-maskTexel.x, 0.0)).r, 0.0, 1.0);
	const float rightMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(maskTexel.x, 0.0)).r, 0.0, 1.0);
	const float topMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(0.0, -maskTexel.y)).r, 0.0, 1.0);
	const float bottomMask = clamp(inputTexture2.sample(quadSampler, maskUV + float2(0.0, maskTexel.y)).r, 0.0, 1.0);
	const float smoothedMask = (centerMask * 4.0 + leftMask + rightMask + topMask + bottomMask) / 8.0;
	const float maskThresholdLow = bool(*isBackCamera) ? 0.62 : 0.5;
	const float maskThresholdHigh = bool(*isBackCamera) ? 0.94 : 0.9;
	const float maskGamma = bool(*isBackCamera) ? 2.0 : 1.7;
	const float personMask = smoothstep(maskThresholdLow, maskThresholdHigh, pow(smoothedMask, maskGamma));

	const half luminance = dot(base.rgb, half3(0.299h, 0.587h, 0.114h));
	const half3 grayscale = half3(luminance, luminance, luminance);
	const half3 rgb = mix(grayscale, base.rgb, half(personMask));
	outputTexture.write(half4(rgb, base.a), gid);
}
