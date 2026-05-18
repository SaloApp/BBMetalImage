#include <metal_stdlib>
using namespace metal;

kernel void dualCameraCompositeKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::read> primaryInput [[texture(1)]],
	texture2d<half, access::sample> secondaryInput [[texture(2)]],
	texture2d<half, access::sample> frameInput [[texture(3)]],
	constant float4 *pipFramePointer [[buffer(0)]],
	constant float4 *frameContentRectPointer [[buffer(1)]],
	constant uint *displayModePointer [[buffer(2)]],
	constant uint *shouldMirrorPrimaryPointer [[buffer(3)]],
	constant uint *shouldMirrorSecondaryPointer [[buffer(4)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	const float4 pipFrame = *pipFramePointer;
	const float4 frameContentRect = *frameContentRectPointer;
	const uint displayMode = *displayModePointer;
	const bool shouldMirrorPrimary = *shouldMirrorPrimaryPointer != 0;
	const bool shouldMirrorSecondary = *shouldMirrorSecondaryPointer != 0;
	const float4 frame = float4(
		float(outputTexture.get_width()) * pipFrame.x,
		float(outputTexture.get_height()) * pipFrame.y,
		float(outputTexture.get_width()) * pipFrame.z,
		float(outputTexture.get_height()) * pipFrame.w
	);

	const uint2 primaryCoordinate = shouldMirrorPrimary
		? uint2(outputTexture.get_width() - 1 - gid.x, gid.y)
		: gid;
	half4 color = primaryInput.read(primaryCoordinate);
	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);

	if (displayMode == 1) {
		const float4 contentFrame = float4(
			frame.x + frame.z * frameContentRect.x,
			frame.y + frame.w * frameContentRect.y,
			frame.z * frameContentRect.z,
			frame.w * frameContentRect.w
		);

		if (
			float(gid.x) >= contentFrame.x &&
			float(gid.x) < contentFrame.x + contentFrame.z &&
			float(gid.y) >= contentFrame.y &&
			float(gid.y) < contentFrame.y + contentFrame.w
		) {
			const float2 contentCoordinate = float2(
				(float(gid.x) - contentFrame.x) / max(contentFrame.z, 1.0),
				(float(gid.y) - contentFrame.y) / max(contentFrame.w, 1.0)
			);
			const float sourceAspect = float(secondaryInput.get_width()) / max(float(secondaryInput.get_height()), 1.0);
			const float destinationAspect = contentFrame.z / max(contentFrame.w, 1.0);
			float2 secondaryCoordinate = contentCoordinate;
			if (sourceAspect > destinationAspect) {
				const float visibleWidth = destinationAspect / max(sourceAspect, 0.0001);
				secondaryCoordinate.x = 0.5 + (contentCoordinate.x - 0.5) * visibleWidth;
			} else {
				const float visibleHeight = sourceAspect / max(destinationAspect, 0.0001);
				secondaryCoordinate.y = 0.5 + (contentCoordinate.y - 0.5) * visibleHeight;
			}
			if (shouldMirrorSecondary) {
				secondaryCoordinate.x = 1.0 - secondaryCoordinate.x;
			}
			color = secondaryInput.sample(quadSampler, secondaryCoordinate);
		}

		if (
			float(gid.x) >= frame.x &&
			float(gid.x) < frame.x + frame.z &&
			float(gid.y) >= frame.y &&
			float(gid.y) < frame.y + frame.w &&
			!(
				float(gid.x) >= contentFrame.x &&
				float(gid.x) < contentFrame.x + contentFrame.z &&
				float(gid.y) >= contentFrame.y &&
				float(gid.y) < contentFrame.y + contentFrame.w
			)
		) {
			const float2 frameCoordinate = float2(
				(float(gid.x) - frame.x) / max(frame.z, 1.0),
				(float(gid.y) - frame.y) / max(frame.w, 1.0)
			);
			const half4 frameColor = frameInput.sample(quadSampler, frameCoordinate);
			color = mix(color, frameColor, frameColor.a);
		}
	} else if (
		float(gid.x) >= frame.x &&
		float(gid.x) < frame.x + frame.z &&
		float(gid.y) >= frame.y &&
		float(gid.y) < frame.y + frame.w
	) {
		const float2 frameCoordinate = float2(
			(float(gid.x) - frame.x) / max(frame.z, 1.0),
			(float(gid.y) - frame.y) / max(frame.w, 1.0)
		);
		const float sourceAspect = float(secondaryInput.get_width()) / max(float(secondaryInput.get_height()), 1.0);
		const float destinationAspect = frame.z / max(frame.w, 1.0);
		float2 secondaryCoordinate = frameCoordinate;
		if (sourceAspect > destinationAspect) {
			const float visibleWidth = destinationAspect / max(sourceAspect, 0.0001);
			secondaryCoordinate.x = 0.5 + (frameCoordinate.x - 0.5) * visibleWidth;
		} else {
			const float visibleHeight = sourceAspect / max(destinationAspect, 0.0001);
			secondaryCoordinate.y = 0.5 + (frameCoordinate.y - 0.5) * visibleHeight;
		}
		if (shouldMirrorSecondary) {
			secondaryCoordinate.x = 1.0 - secondaryCoordinate.x;
		}
		color = secondaryInput.sample(quadSampler, secondaryCoordinate);
	}

	outputTexture.write(color, gid);
}
