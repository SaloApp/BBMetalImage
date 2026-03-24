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
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	float s = clamp(*strength, 0.0f, 1.0f);
	float lc = clamp(*localContrast, 0.0f, 1.0f);

	half4 base = inputTexture.read(gid);
	float3 color = float3(base.rgb);
	float luma = max(luminance(color), 1e-5f);

	// Lift darker regions and compress highlights.
	float shadowLift = smoothstep(0.45f, 0.02f, luma) * (0.28f * s);
	float highlightRollOff = smoothstep(0.55f, 1.0f, luma) * (0.24f * s);
	float targetLuma = clamp(luma + shadowLift - highlightRollOff, 0.0f, 1.0f);
	float3 mapped = color * (targetLuma / luma);

	// Local contrast approximation from 4-neighborhood luma.
	uint2 left = uint2(gid.x > 0 ? gid.x - 1 : gid.x, gid.y);
	uint2 right = uint2(min(gid.x + 1, outputTexture.get_width() - 1), gid.y);
	uint2 up = uint2(gid.x, gid.y > 0 ? gid.y - 1 : gid.y);
	uint2 down = uint2(gid.x, min(gid.y + 1, outputTexture.get_height() - 1));

	float neighborAvg = (
		luminance(float3(inputTexture.read(left).rgb)) +
		luminance(float3(inputTexture.read(right).rgb)) +
		luminance(float3(inputTexture.read(up).rgb)) +
		luminance(float3(inputTexture.read(down).rgb))
	) * 0.25f;

	float localDelta = (luma - neighborAvg) * (0.35f * lc * s);
	mapped += localDelta;

	// Gentle tone shaping to keep output stable.
	float3 compressed = mapped / (1.0f + 0.35f * s * mapped);
	float3 finalColor = mix(mapped, compressed, 0.6f * s);

	outputTexture.write(half4(half3(clamp(finalColor, 0.0f, 1.0f)), base.a), gid);
}
