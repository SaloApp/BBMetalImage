#include <metal_stdlib>
using namespace metal;

struct AnalogGrainParams {
	float amount;
	float grainSize;
	float lumaAmount;
	float chromaAmount;
	float time;
	float seed;
};

float analogGrainHash(float2 p) {
	return fract(sin(dot(p, float2(127.1, 311.7))) * 43758.5453123);
}

float analogGrainModifier(float2 pixel, float frameSeed, float seed) {
	return analogGrainHash(pixel + float2(seed + frameSeed, seed * 1.73 - frameSeed)) * 2.0 - 1.0;
}

float3 analogGrainPerturbation(float2 pixel, float grainPixels, float frameSeed, constant AnalogGrainParams &params) {
	float2 grainPixel = floor(pixel / max(grainPixels, 1.0));
	float lumaModifier = analogGrainModifier(grainPixel, frameSeed, params.seed);
	float3 channelModifiers = float3(
		analogGrainModifier(grainPixel + float2(17.0, 3.0), frameSeed, params.seed + 11.0),
		analogGrainModifier(grainPixel + float2(5.0, 23.0), frameSeed, params.seed + 37.0),
		analogGrainModifier(grainPixel + float2(29.0, 13.0), frameSeed, params.seed + 71.0)
	);

	return float3(lumaModifier) * clamp(params.lumaAmount, 0.0, 1.0)
		+ channelModifiers * clamp(params.chromaAmount, 0.0, 1.0);
}

kernel void analogGrainKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant AnalogGrainParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }

	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 pixel = float2(gid);
	float2 uv = (pixel + 0.5) / resolution;

	float4 sampled = float4(inputTexture.sample(quadSampler, uv));
	float grainSize = clamp(params.grainSize, 0.0, 1.0);
	float grainPixels = mix(1.6, 5.0, grainSize);
	float fuzzPixels = mix(1.0, 5.5, grainSize);
	float2 fuzzyUV = clamp((pixel + float2(fuzzPixels * 1.8, fuzzPixels + 1.0) + 0.5) / resolution, 0.0, 1.0);
	float3 fuzzyColor = float3(inputTexture.sample(quadSampler, fuzzyUV).rgb);

	float amount = clamp(params.amount, 0.0, 1.0);
	float fuzzMix = amount * mix(0.18, 0.36, grainSize);
	float3 baseColor = mix(sampled.rgb, (sampled.rgb + fuzzyColor) * 0.5, fuzzMix);

	float frameSeed = floor(params.time * 12.0) * 0.173;
	float3 perturbation = analogGrainPerturbation(pixel, grainPixels, frameSeed, params);
	float3 finalColor = clamp(baseColor + perturbation * amount * 0.145, 0.0, 1.0);

	outputTexture.write(half4(half3(finalColor), half(sampled.a)), gid);
}
