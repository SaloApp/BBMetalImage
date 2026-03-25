#include <metal_stdlib>
using namespace metal;

static inline float luminance(float3 color) {
	return dot(color, float3(0.2126, 0.7152, 0.0722));
}

kernel void wdrKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::read> inputTexture [[texture(1)]],
	constant float *strength [[buffer(0)]],
	constant float *localContrast [[buffer(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	const uint outputWidth = outputTexture.get_width();
	const uint outputHeight = outputTexture.get_height();
	const uint inputWidth = inputTexture.get_width();
	const uint inputHeight = inputTexture.get_height();

	if (gid.x >= outputWidth || gid.y >= outputHeight || gid.x >= inputWidth || gid.y >= inputHeight) {
		return;
	}

	float s = clamp(*strength, 0.0f, 1.0f);
	float lc = clamp(*localContrast, 0.0f, 1.0f);

	half4 base = inputTexture.read(gid);
	float3 color = clamp(float3(base.rgb), 0.0f, 1.0f);
	float luma = clamp(luminance(color), 0.0f, 1.0f);

	// Lift darker regions and compress highlights.
	float shadowMask = 1.0f - smoothstep(0.02f, 0.45f, luma);
	float shadowLift = shadowMask * (0.28f * s);
	float highlightRollOff = smoothstep(0.55f, 1.0f, luma) * (0.24f * s);
	float targetLuma = clamp(luma + shadowLift - highlightRollOff, 0.0f, 1.0f);

	// Prevent near-black chroma spikes by stabilizing chroma reconstruction in deep shadows.
	float safeLuma = max(luma, 0.03f);
	float3 chromaRatio = clamp(color / safeLuma, 0.0f, 4.0f);
	float shadowChromaKeep = mix(0.35f, 1.0f, smoothstep(0.04f, 0.22f, luma));
	float3 mappedNeutral = float3(targetLuma);
	float3 mappedColor = clamp(chromaRatio * targetLuma, 0.0f, 1.0f);
	float3 mapped = mix(mappedNeutral, mappedColor, shadowChromaKeep);

	// Local contrast approximation from 4-neighborhood luma.
	uint2 left = uint2(gid.x > 0 ? gid.x - 1 : gid.x, gid.y);
	uint2 right = uint2(min(gid.x + 1, inputWidth - 1), gid.y);
	uint2 up = uint2(gid.x, gid.y > 0 ? gid.y - 1 : gid.y);
	uint2 down = uint2(gid.x, min(gid.y + 1, inputHeight - 1));

	float neighborAvg = (
		luminance(float3(inputTexture.read(left).rgb)) +
		luminance(float3(inputTexture.read(right).rgb)) +
		luminance(float3(inputTexture.read(up).rgb)) +
		luminance(float3(inputTexture.read(down).rgb))
	) * 0.25f;

	float localShadowAttenuation = mix(0.25f, 1.0f, smoothstep(0.03f, 0.20f, luma));
	float localDelta = (luma - neighborAvg) * (0.35f * lc * s * localShadowAttenuation);
	mapped = clamp(mapped + localDelta, 0.0f, 1.0f);

	// Gentle luma-based tone shaping to keep hue stable.
	float mappedLuma = clamp(luminance(mapped), 0.0f, 1.0f);
	float compression = 1.0f / max(1e-3f, 1.0f + 0.35f * s * mappedLuma);
	float3 compressed = mapped * compression;
	float3 finalColor = clamp(mix(mapped, compressed, 0.6f * s), 0.0f, 1.0f);

	outputTexture.write(half4(half3(finalColor), base.a), gid);
}
