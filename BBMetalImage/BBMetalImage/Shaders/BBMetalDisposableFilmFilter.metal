#include <metal_stdlib>
using namespace metal;

struct DisposableFilmParams {
	float exposure;
	float contrast;
	float shadows;
	float highlights;
	float temperature;
	float tint;
	float saturation;
	float skinTone;
	float fade;
	float greenShadow;
};

float3 disposableRGBToHSV(float3 c) {
	float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
	float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 disposableHSVToRGB(float3 c) {
	float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float disposableHueWeight(float hue, float center, float width) {
	float distance = abs(fract(hue - center + 0.5) - 0.5);
	return 1.0 - smoothstep(width * 0.45, width, distance);
}

kernel void disposableFilmKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant DisposableFilmParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }

	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float4 sampled = float4(inputTexture.sample(quadSampler, uv));
	float3 color = sampled.rgb;

	color *= exp2(clamp(params.exposure, -3.0, 3.0) * 0.38);

	float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	float shadowMask = 1.0 - smoothstep(0.08, 0.58, luma);
	float highlightMask = smoothstep(0.40, 0.90, luma);
	float greenShadow = clamp(params.greenShadow, -8.0, 8.0) * 0.010;

	color += shadowMask * clamp(params.shadows, -8.0, 8.0) * 0.018;
	color += highlightMask * clamp(params.highlights, -8.0, 8.0) * 0.012;
	color += shadowMask * float3(-greenShadow * 0.36, greenShadow, -greenShadow * 0.22);

	float temperature = clamp(params.temperature, -8.0, 8.0) * 0.018;
	float tint = clamp(params.tint, -8.0, 8.0) * 0.014;
	color *= float3(
		1.0 + temperature + tint * 0.18,
		1.0 - tint * 0.28,
		1.0 - temperature * 0.62 + tint * 0.30
	);
	color += float3(tint * 0.40, -tint * 0.28, tint * 0.24);

	float contrastScale = 1.0 + clamp(params.contrast, -5.0, 5.0) * 0.12;
	color = (color - 0.42) * contrastScale + 0.42;

	float3 hsv = disposableRGBToHSV(max(color, 0.0));
	float redWeight = disposableHueWeight(hsv.x, 0.0, 0.075) + disposableHueWeight(hsv.x, 1.0, 0.075);
	float orangeWeight = disposableHueWeight(hsv.x, 0.080, 0.085);
	float yellowWeight = disposableHueWeight(hsv.x, 0.155, 0.080);
	float greenWeight = disposableHueWeight(hsv.x, 0.330, 0.120);
	float blueWeight = disposableHueWeight(hsv.x, 0.600, 0.120);
	float warmWeight = clamp(redWeight + orangeWeight, 0.0, 1.0);

	float skinLuma = smoothstep(0.16, 0.72, luma) * (1.0 - smoothstep(0.88, 1.0, luma));
	float skinSaturation = smoothstep(0.05, 0.55, hsv.y);
	float skinMask = clamp(warmWeight * skinLuma * skinSaturation, 0.0, 1.0);
	float skinWarmth = clamp(-params.skinTone, -5.0, 5.0) * 0.012;
	color = mix(
		color,
		color * float3(1.0 + skinWarmth * 0.70, 1.0 + skinWarmth * 0.06, 1.0 - skinWarmth * 0.45) + float3(skinWarmth * 0.22, skinWarmth * 0.04, -skinWarmth * 0.08),
		skinMask * 0.55
	);

	hsv = disposableRGBToHSV(max(color, 0.0));
	redWeight = disposableHueWeight(hsv.x, 0.0, 0.075) + disposableHueWeight(hsv.x, 1.0, 0.075);
	orangeWeight = disposableHueWeight(hsv.x, 0.080, 0.085);
	yellowWeight = disposableHueWeight(hsv.x, 0.155, 0.080);
	greenWeight = disposableHueWeight(hsv.x, 0.330, 0.120);
	blueWeight = disposableHueWeight(hsv.x, 0.600, 0.120);
	float totalWeight = max(redWeight + orangeWeight + yellowWeight + greenWeight + blueWeight, 0.001);
	float hueShift = (
		redWeight * -0.003 +
		orangeWeight * -0.004 +
		yellowWeight * -0.010 +
		greenWeight * -0.006 +
		blueWeight * 0.008
	) / totalWeight;
	float saturationCurve = (
		(redWeight + orangeWeight) * 0.10 +
		yellowWeight * -0.05 +
		greenWeight * -0.16 +
		blueWeight * -0.10
	) / totalWeight;
	hsv.x = fract(hsv.x + hueShift + 1.0);
	hsv.y *= clamp(1.0 + saturationCurve, 0.20, 1.30);
	hsv.z += (greenWeight + blueWeight) * 0.012;
	color = disposableHSVToRGB(hsv);

	luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	float saturationScale = 1.0 + clamp(params.saturation, -5.0, 5.0) * 0.08;
	color = mix(float3(luma), color, clamp(saturationScale, 0.0, 1.35));

	float fadeAmount = clamp(params.fade, 0.0, 8.0) * 0.035;
	color = mix(color, color * 0.86 + float3(0.098, 0.110, 0.095), fadeAmount);

	float2 centered = (uv - 0.5) * float2(resolution.x / max(resolution.y, 1.0), 1.0);
	float vignette = 1.0 - smoothstep(0.16, 0.72, dot(centered, centered));
	color *= mix(0.80, 1.0, vignette);
	color = mix(color, color * float3(0.95, 1.03, 0.96), (1.0 - vignette) * 0.22);

	color = max(color, float3(0.010 + fadeAmount * 0.10));
	color = color / (color + float3(0.18));
	color *= 0.92;
	color = pow(clamp(color, 0.0, 1.0), float3(0.96));

	outputTexture.write(half4(half3(clamp(color, 0.0, 1.0)), half(sampled.a)), gid);
}
