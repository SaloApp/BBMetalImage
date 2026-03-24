#include <metal_stdlib>
using namespace metal;

kernel void binaryLUT3DKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::read> inputTexture [[texture(1)]],
	texture3d<half, access::sample> lutTexture [[texture(2)]],
	constant float *intensity [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	const half4 baseColor = inputTexture.read(gid);
	constexpr sampler lutSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
	const half3 lutColor = lutTexture.sample(lutSampler, float3(baseColor.rgb)).rgb;
	const half mixAmount = half(clamp(*intensity, 0.0f, 1.0f));
	const half3 outputColor = mix(baseColor.rgb, lutColor, mixAmount);

	outputTexture.write(half4(outputColor, baseColor.a), gid);
}
