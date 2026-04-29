#include <metal_stdlib>
using namespace metal;

struct PhotoboothParams {
	float filledMask;
	float lineWidth;
	float contrast;
	float brightness;
};

float4 photoboothSample(
	int slot,
	float2 uv,
	texture2d<half, access::sample> frame0,
	texture2d<half, access::sample> frame1,
	texture2d<half, access::sample> frame2,
	texture2d<half, access::sample> frame3
) {
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	if (slot == 0) { return float4(frame0.sample(quadSampler, uv)); }
	if (slot == 1) { return float4(frame1.sample(quadSampler, uv)); }
	if (slot == 2) { return float4(frame2.sample(quadSampler, uv)); }
	return float4(frame3.sample(quadSampler, uv));
}

float photoboothMonochrome(float3 color, constant PhotoboothParams &params) {
	float luma = dot(color, float3(0.2126, 0.7152, 0.0722));
	luma = pow(clamp(luma, 0.0, 1.0), 0.86);
	luma = clamp((luma - 0.47) * params.contrast + 0.52 + params.brightness, 0.0, 1.0);
	return smoothstep(0.02, 0.98, luma);
}

kernel void photoboothKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	texture2d<half, access::sample> frame0 [[texture(2)]],
	texture2d<half, access::sample> frame1 [[texture(3)]],
	texture2d<half, access::sample> frame2 [[texture(4)]],
	texture2d<half, access::sample> frame3 [[texture(5)]],
	constant PhotoboothParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }

	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 pixel = float2(gid) + 0.5;
	float lineWidthPixels = max(params.lineWidth * min(resolution.x, resolution.y), 2.0);
	float halfLineWidth = lineWidthPixels * 0.5;
	float2 center = resolution * 0.5;

	bool outerLine = pixel.x <= lineWidthPixels ||
		pixel.y <= lineWidthPixels ||
		pixel.x >= resolution.x - lineWidthPixels ||
		pixel.y >= resolution.y - lineWidthPixels;
	bool innerLine = abs(pixel.x - center.x) <= halfLineWidth ||
		abs(pixel.y - center.y) <= halfLineWidth;
	if (outerLine || innerLine) {
		outputTexture.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
		return;
	}

	float2 uv = pixel / resolution;
	bool right = uv.x >= 0.5;
	bool bottom = uv.y >= 0.5;
	int slot = bottom ? (right ? 2 : 3) : (right ? 1 : 0);
	int filledMask = int(params.filledMask + 0.5);
	if ((filledMask & (1 << slot)) == 0) {
		outputTexture.write(half4(0.0h, 0.0h, 0.0h, 1.0h), gid);
		return;
	}

	float2 cellUV = float2(
		right ? (uv.x - 0.5) * 2.0 : uv.x * 2.0,
		bottom ? (uv.y - 0.5) * 2.0 : uv.y * 2.0
	);
	cellUV = clamp(cellUV, 0.0, 1.0);

	float4 sampled = photoboothSample(slot, cellUV, frame0, frame1, frame2, frame3);
	float mono = photoboothMonochrome(sampled.rgb, params);
	float3 outputColor = float3(mono);

	outputTexture.write(half4(half3(outputColor), half(sampled.a)), gid);
}
