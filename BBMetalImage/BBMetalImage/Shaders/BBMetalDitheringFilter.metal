#include <metal_stdlib>
using namespace metal;

struct DitheringParams {
	float4 darkColor;
	float4 lightColor;
	float amount;
	float pixelSize;
	float patternType;
	float contrast;
	float time;
	float seed;
};

float ditheringHash21(float2 p) {
	p = fract(p * float2(123.34, 456.21));
	p += dot(p, p + 45.32);
	return fract(p.x * p.y);
}

float ditheringBayer2x2(int2 cell) {
	int x = cell.x & 1;
	int y = cell.y & 1;
	int index = y * 2 + x;
	int value = index == 0 ? 0 : (index == 1 ? 2 : (index == 2 ? 3 : 1));
	return (float(value) + 0.5) / 4.0;
}

float ditheringBayer4x4(int2 cell) {
	const int values[16] = {
		0, 8, 2, 10,
		12, 4, 14, 6,
		3, 11, 1, 9,
		15, 7, 13, 5
	};
	int x = cell.x & 3;
	int y = cell.y & 3;
	return (float(values[y * 4 + x]) + 0.5) / 16.0;
}

float ditheringBayer8x8(int2 cell) {
	const int values[64] = {
		0, 32, 8, 40, 2, 34, 10, 42,
		48, 16, 56, 24, 50, 18, 58, 26,
		12, 44, 4, 36, 14, 46, 6, 38,
		60, 28, 52, 20, 62, 30, 54, 22,
		3, 35, 11, 43, 1, 33, 9, 41,
		51, 19, 59, 27, 49, 17, 57, 25,
		15, 47, 7, 39, 13, 45, 5, 37,
		63, 31, 55, 23, 61, 29, 53, 21
	};
	int x = cell.x & 7;
	int y = cell.y & 7;
	return (float(values[y * 8 + x]) + 0.5) / 64.0;
}

float ditheringThreshold(int2 cell, constant DitheringParams &params) {
	int type = int(floor(params.patternType + 0.5));
	if (type == 1) {
		float frame = floor(params.time * 8.0) * 31.0;
		return ditheringHash21(float2(cell) + frame + params.seed);
	}
	if (type == 2) {
		return ditheringBayer2x2(cell);
	}
	if (type == 3) {
		return ditheringBayer4x4(cell);
	}
	return ditheringBayer8x8(cell);
}

kernel void ditheringKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant DitheringParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }

	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 pixel = float2(gid);
	float pixelSize = clamp(params.pixelSize, 1.0, 18.0);
	float2 ditherCell = floor(pixel / pixelSize);
	float2 cellCenter = (ditherCell + 0.5) * pixelSize;
	float2 uv = clamp((cellCenter + 0.5) / resolution, 0.0, 1.0);

	float4 sampled = float4(inputTexture.sample(quadSampler, uv));
	float luma = dot(sampled.rgb, float3(0.2126, 0.7152, 0.0722));
	float contrast = 1.0 + clamp(params.contrast, 0.0, 1.0) * 3.0;
	luma = pow(clamp(luma, 0.0, 1.0), 0.82);
	luma = clamp((luma - 0.48) * contrast + 0.50, 0.0, 1.0);

	float threshold = ditheringThreshold(int2(ditherCell), params);
	float thresholdNoise = (ditheringHash21(ditherCell + params.seed) - 0.5) * 0.045;

	float2 cellLocal = fract(pixel / pixelSize);
	float innerX = smoothstep(0.10, 0.22, cellLocal.x) * (1.0 - smoothstep(0.78, 0.92, cellLocal.x));
	float innerY = smoothstep(0.08, 0.18, cellLocal.y) * (1.0 - smoothstep(0.78, 0.92, cellLocal.y));
	float pixelMask = mix(0.62, 1.0, innerX * innerY);
	float scanline = mix(0.86, 1.0, step(0.5, fract(pixel.y * 0.5)));
	float flicker = 0.97 + 0.03 * ditheringHash21(ditherCell + floor(params.time * 10.0) + params.seed);
	float3 litGreen = params.lightColor.rgb * pixelMask * scanline * flicker;
	float3 ditherColor = mix(params.darkColor.rgb, litGreen, step(threshold + thresholdNoise, luma));

	float amount = clamp(params.amount, 0.0, 1.0);
	float3 finalColor = clamp(mix(sampled.rgb, ditherColor, amount), 0.0, 1.0);

	outputTexture.write(half4(half3(finalColor), half(sampled.a)), gid);
}
