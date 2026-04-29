#include <metal_stdlib>
using namespace metal;

struct NinetiesParams {
	float exposure;
	float contrast;
	float shadows;
	float saturation;
	float clarity;
	float fade;
};

float3 ninetiesSample(
	texture2d<half, access::sample> inputTexture,
	sampler quadSampler,
	float2 uv
) {
	return float3(inputTexture.sample(quadSampler, clamp(uv, 0.0, 1.0)).rgb);
}

kernel void ninetiesKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant NinetiesParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }

	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float2 texel = 1.0 / resolution;

	float3 color = ninetiesSample(inputTexture, quadSampler, uv);
	float3 blur = color * 0.36;
	blur += ninetiesSample(inputTexture, quadSampler, uv + float2(texel.x * 2.0, 0.0)) * 0.13;
	blur += ninetiesSample(inputTexture, quadSampler, uv - float2(texel.x * 2.0, 0.0)) * 0.13;
	blur += ninetiesSample(inputTexture, quadSampler, uv + float2(0.0, texel.y * 2.0)) * 0.13;
	blur += ninetiesSample(inputTexture, quadSampler, uv - float2(0.0, texel.y * 2.0)) * 0.13;
	blur += ninetiesSample(inputTexture, quadSampler, uv + texel * float2(1.5, 1.5)) * 0.06;
	blur += ninetiesSample(inputTexture, quadSampler, uv - texel * float2(1.5, 1.5)) * 0.06;

	float exposureScale = exp2(clamp(params.exposure, -3.0, 3.0) * 0.45);
	color *= exposureScale;

	float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	float shadowMask = 1.0 - smoothstep(0.08, 0.58, luma);
	float shadowLift = clamp(params.shadows, -10.0, 10.0) * 0.016;
	color += shadowMask * shadowLift;

	float clarity = clamp(params.clarity, -5.0, 5.0) * 0.14;
	color += (color - blur * exposureScale) * clarity;

	float contrastScale = 1.0 + clamp(params.contrast, -5.0, 5.0) * 0.15;
	float pivot = 0.46;
	color = (color - pivot) * contrastScale + pivot;

	float fadeAmount = clamp(params.fade, 0.0, 10.0) * 0.035;
	color = mix(color, color * 0.82 + float3(0.105, 0.105, 0.095), fadeAmount);

	luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	float saturationScale = max(0.0, 1.0 + clamp(params.saturation, -5.0, 5.0) * 0.34);
	color = mix(float3(luma), color, saturationScale);

	float cooledShadow = 1.0 - smoothstep(0.16, 0.62, luma);
	float warmHighlight = smoothstep(0.48, 0.94, luma);
	color = mix(color, color * float3(0.90, 0.96, 1.05), cooledShadow * 0.22);
	color = mix(color, color * float3(1.05, 1.01, 0.92) + float3(0.012, 0.008, 0.0), warmHighlight * 0.18);

	color = clamp(color, 0.0, 1.0);
	color = pow(color, float3(0.94));
	color = color / (color + float3(0.12));
	color *= 1.12;

	float2 vignetteUV = uv * (1.0 - uv.yx);
	float vignette = smoothstep(0.02, 0.32, vignetteUV.x * vignetteUV.y * 5.0);
	color *= mix(0.88, 1.0, vignette);

	float alpha = float(inputTexture.sample(quadSampler, uv).a);
	outputTexture.write(half4(half3(clamp(color, 0.0, 1.0)), half(alpha)), gid);
}
