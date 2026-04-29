#include <metal_stdlib>
using namespace metal;

struct Fuji400HParams {
	float exposure;
	float contrast;
	float highlights;
	float shadows;
	float whites;
	float blacks;
	float warmth;
	float tint;
	float vibrance;
	float saturation;
};

float3 fujiRGBToHSV(float3 c) {
	float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
	float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

float3 fujiHSVToRGB(float3 c) {
	float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float fujiHueWeight(float hue, float center, float width) {
	float distance = abs(fract(hue - center + 0.5) - 0.5);
	return 1.0 - smoothstep(width * 0.45, width, distance);
}

float3 fujiColorGradeTint(float hueDegrees, float saturation) {
	return fujiHSVToRGB(float3(hueDegrees / 360.0, saturation, 1.0));
}

kernel void fuji400HKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant Fuji400HParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }

	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float4 sampled = float4(inputTexture.sample(quadSampler, uv));
	float3 color = sampled.rgb;

	color *= exp2(clamp(params.exposure, -3.0, 3.0) * 0.72);
	float luma = dot(color, float3(0.2126, 0.7152, 0.0722));

	float shadowMask = 1.0 - smoothstep(0.08, 0.58, luma);
	float blackMask = 1.0 - smoothstep(0.02, 0.34, luma);
	float highlightMask = smoothstep(0.48, 0.92, luma);
	float whiteMask = smoothstep(0.74, 1.0, luma);

	color += shadowMask * clamp(params.shadows, -100.0, 100.0) * 0.0025;
	color += blackMask * clamp(params.blacks, -100.0, 100.0) * 0.0018;
	color += highlightMask * clamp(params.highlights, -100.0, 100.0) * 0.0022;
	color += whiteMask * clamp(params.whites, -100.0, 100.0) * 0.0020;

	float contrastScale = 1.0 + clamp(params.contrast, -100.0, 100.0) * 0.0075;
	color = (color - 0.45) * contrastScale + 0.45;

	float warmth = clamp(params.warmth, -100.0, 100.0) * 0.0014;
	float tint = clamp(params.tint, -100.0, 100.0) * 0.0012;
	color *= float3(1.0 + warmth, 1.0 - tint * 0.6, 1.0 - warmth + tint * 0.45);
	color += float3(tint * 0.45, -tint * 0.35, tint * 0.28);

	float3 hsv = fujiRGBToHSV(max(color, 0.0));
	float redWeight = fujiHueWeight(hsv.x, 0.0, 0.070) + fujiHueWeight(hsv.x, 1.0, 0.070);
	float orangeWeight = fujiHueWeight(hsv.x, 0.080, 0.075);
	float yellowWeight = fujiHueWeight(hsv.x, 0.160, 0.080);
	float greenWeight = fujiHueWeight(hsv.x, 0.330, 0.120);
	float blueWeight = fujiHueWeight(hsv.x, 0.590, 0.120);
	float purpleWeight = fujiHueWeight(hsv.x, 0.760, 0.085);
	float magentaWeight = fujiHueWeight(hsv.x, 0.875, 0.090);
	float totalWeight = max(
		redWeight + orangeWeight + yellowWeight + greenWeight + blueWeight + purpleWeight + magentaWeight,
		0.001
	);

	float hueShift = (
		redWeight * 0.0045 +
		orangeWeight * -0.0035 +
		yellowWeight * -0.0180 +
		greenWeight * -0.0300 +
		blueWeight * -0.0080
	) / totalWeight;
	hsv.x = fract(hsv.x + hueShift + 1.0);

	float hslSatAdjustment = (
		redWeight * -0.15 +
		orangeWeight * -0.12 +
		yellowWeight * -0.35 +
		greenWeight * -0.45 +
		blueWeight * -0.30 +
		purpleWeight * -0.80 +
		magentaWeight * -0.80
	) / totalWeight;
	hsv.y *= clamp(1.0 + hslSatAdjustment * 0.85, 0.08, 1.35);

	float hslLuminance = (
		redWeight * 0.05 +
		orangeWeight * 0.14 +
		yellowWeight * 0.05 +
		greenWeight * 0.10 +
		blueWeight * 0.10
	) / totalWeight;
	hsv.z = clamp(hsv.z + hslLuminance * 0.18, 0.0, 1.0);
	color = fujiHSVToRGB(hsv);

	luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	float globalSaturation = 1.0 + clamp(params.saturation, -100.0, 100.0) * 0.0042;
	float vibranceMask = 1.0 - smoothstep(0.32, 0.82, hsv.y);
	float vibranceScale = 1.0 + clamp(params.vibrance, -100.0, 100.0) * 0.0032 * vibranceMask;
	color = mix(float3(luma), color, clamp(globalSaturation * vibranceScale, 0.0, 1.25));

	luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	float shadowGrade = 1.0 - smoothstep(0.12, 0.54, luma);
	float midtoneGrade = 1.0 - smoothstep(0.18, 0.48, abs(luma - 0.50));
	float highlightGrade = smoothstep(0.48, 0.92, luma);
	float3 shadowTint = fujiColorGradeTint(215.0, 0.06);
	float3 midtoneTint = fujiColorGradeTint(35.0, 0.05);
	float3 highlightTint = fujiColorGradeTint(50.0, 0.08);

	color = mix(color, color * shadowTint, shadowGrade * 0.22);
	color = mix(color, color * midtoneTint, midtoneGrade * 0.16);
	color = mix(color, color * highlightTint + float3(0.010, 0.008, 0.002), highlightGrade * 0.18);

	float matteLift = 0.018 + shadowMask * 0.018;
	color = max(color, float3(matteLift));
	color = clamp(color, 0.0, 1.0);
	color = pow(color, float3(0.96));

	outputTexture.write(half4(half3(clamp(color, 0.0, 1.0)), half(sampled.a)), gid);
}
