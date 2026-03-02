#include <metal_stdlib>
using namespace metal;

kernel void camcorderCompositeKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	texture2d<half, access::sample> inputTexture2 [[texture(2)]],
	texture2d<half, access::sample> inputTexture3 [[texture(3)]],
	constant float4 *viewfinderRect [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
	const float width = float(outputTexture.get_width());
	const float height = float(outputTexture.get_height());
	const float2 uv = float2(
		(float(gid.x) + 0.5) / width,
		(float(gid.y) + 0.5) / height
	);

	const half4 blurredBase = inputTexture.sample(quadSampler, uv);

	const float outAspect = width / max(height, 1.0);
	const float assetWidth = float(inputTexture3.get_width());
	const float assetHeight = float(inputTexture3.get_height());
	const float assetAspect = assetWidth / max(assetHeight, 1.0);

	float2 assetUV = uv;
	float assetScaleX = 1.0;
	float assetScaleY = 1.0;
	if (assetAspect > outAspect) {
		// Aspect fill: source is wider, so crop left/right.
		const float normalizedWidth = outAspect / assetAspect;
		assetUV.x = (uv.x - 0.5) * normalizedWidth + 0.5;
		assetScaleX = normalizedWidth;
	} else if (assetAspect < outAspect) {
		// Aspect fill: source is taller, so crop top/bottom.
		const float normalizedHeight = assetAspect / outAspect;
		assetUV.y = (uv.y - 0.5) * normalizedHeight + 0.5;
		assetScaleY = normalizedHeight;
	}

	half4 overlay = half4(0.0h);
	if (assetUV.x >= 0.0 && assetUV.x <= 1.0 && assetUV.y >= 0.0 && assetUV.y <= 1.0) {
		overlay = inputTexture3.sample(quadSampler, assetUV);
	}

	const half backgroundVisibility = 0.8h;
	const half3 darkenedBackground = blurredBase.rgb * backgroundVisibility;
	half3 color = mix(darkenedBackground, overlay.rgb, overlay.a);

	const float2 rawViewfinderMin = float2((*viewfinderRect).x, (*viewfinderRect).y);
	const float2 rawViewfinderMax = float2((*viewfinderRect).z, (*viewfinderRect).w);
	const float2 viewfinderMin = clamp(min(rawViewfinderMin, rawViewfinderMax), float2(0.0), float2(1.0));
	const float2 viewfinderMax = clamp(max(rawViewfinderMin, rawViewfinderMax), float2(0.0), float2(1.0));
	const float2 viewfinderSize = max(viewfinderMax - viewfinderMin, float2(1e-6));

	const bool isInsideInset =
		assetUV.x >= viewfinderMin.x && assetUV.x <= viewfinderMax.x &&
		assetUV.y >= viewfinderMin.y && assetUV.y <= viewfinderMax.y;

	if (isInsideInset) {
		const float2 localUV = clamp((assetUV - viewfinderMin) / viewfinderSize, float2(0.0), float2(1.0));
		// Viewfinder aspect in output pixel space after overlay aspect-fill scaling.
		const float viewfinderWidthInOutput = (viewfinderSize.x / max(assetScaleX, 1e-6)) * width;
		const float viewfinderHeightInOutput = (viewfinderSize.y / max(assetScaleY, 1e-6)) * height;
		const float viewfinderAspect = viewfinderWidthInOutput / max(viewfinderHeightInOutput, 1e-6);
		const float cameraAspect = float(inputTexture2.get_width()) / max(float(inputTexture2.get_height()), 1.0);

		float2 insetUV = localUV;
		if (cameraAspect > viewfinderAspect) {
			// Camera is wider: crop left/right for aspect-fill.
			const float normalizedWidth = viewfinderAspect / cameraAspect;
			insetUV.x = (localUV.x - 0.5) * normalizedWidth + 0.5;
		} else if (cameraAspect < viewfinderAspect) {
			// Camera is taller: crop top/bottom for aspect-fill.
			const float normalizedHeight = cameraAspect / viewfinderAspect;
			insetUV.y = (localUV.y - 0.5) * normalizedHeight + 0.5;
		}

		const half4 inset = inputTexture2.sample(quadSampler, clamp(insetUV, float2(0.0), float2(1.0)));
		color = inset.rgb;
	}

	outputTexture.write(half4(color, blurredBase.a), gid);
}
