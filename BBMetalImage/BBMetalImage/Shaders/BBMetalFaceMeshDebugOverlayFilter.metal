#include <metal_stdlib>
using namespace metal;

// Debug visualiser: the landmark set drawn straight onto the frame, matching what the official
// MediaPipe iOS sample strokes — dots for every point, polylines for the face oval, brows, eyes and
// lips.
//
// This exists to answer one question that nothing else can: did the landmarks land where we think
// they did? Index tables, UV conventions and the Y direction are all things that go wrong silently
// and then get mistaken for a badly tuned preset.
//
// 478 points is 3824 bytes and 108 segments is 864 — both under the 4KB `setBytes` ceiling, so
// these travel as constant data with no buffer to allocate, version or keep alive across frames.
// The loop is confined to the padded face box.

static inline float distanceToSegment(float2 point, float2 start, float2 end) {
	const float2 span = end - start;
	const float lengthSquared = dot(span, span);
	if (lengthSquared < 1e-12) {
		return length(point - start);
	}
	const float t = clamp(dot(point - start, span) / lengthSquared, 0.0, 1.0);
	return length(point - (start + span * t));
}

kernel void faceMeshDebugOverlayKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float2 *points [[buffer(0)]],
	constant uint *pointCountPtr [[buffer(1)]],
	constant uint2 *connections [[buffer(2)]],
	constant uint *connectionCountPtr [[buffer(3)]],
	constant float4 *boundsPtr [[buffer(4)]],
	constant float2 *sizesPtr [[buffer(5)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
		return;
	}

	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
	const float width = float(outputTexture.get_width());
	const float height = float(outputTexture.get_height());
	const float2 uv = float2((float(gid.x) + 0.5) / width, (float(gid.y) + 0.5) / height);
	const half4 base = inputTexture.sample(quadSampler, uv);

	const uint pointCount = *pointCountPtr;
	const uint connectionCount = *connectionCountPtr;
	const float4 bounds = *boundsPtr;
	const bool insideBounds =
		uv.x >= bounds.x && uv.x <= bounds.z && uv.y >= bounds.y && uv.y <= bounds.w;
	if (pointCount == 0u || !insideBounds) {
		outputTexture.write(base, gid);
		return;
	}

	// Distances are measured in a square space so dots stay round and strokes stay even on a
	// non-square frame.
	const float aspect = width / max(height, 1.0);
	const float2 correction = float2(aspect, 1.0);
	const float dotRadius = sizesPtr->x;
	const float lineWidth = sizesPtr->y;
	const float2 here = uv * correction;

	float nearestLine = 1e6;
	for (uint i = 0u; i < connectionCount; ++i) {
		const uint2 pair = connections[i];
		if (pair.x >= pointCount || pair.y >= pointCount) {
			continue;
		}
		nearestLine = min(
			nearestLine,
			distanceToSegment(here, points[pair.x] * correction, points[pair.y] * correction)
		);
	}

	float nearestPoint = 1e6;
	for (uint i = 0u; i < pointCount; ++i) {
		nearestPoint = min(nearestPoint, length(here - points[i] * correction));
	}

	// Antialias over roughly one pixel so the overlay reads at preview resolution.
	const float feather = 1.5 / max(width, 1.0);
	const float lineAlpha = 1.0 - smoothstep(lineWidth - feather, lineWidth + feather, nearestLine);
	const float dotAlpha = 1.0 - smoothstep(dotRadius - feather, dotRadius + feather, nearestPoint);

	const half3 lineColour = half3(0.25h, 0.85h, 1.0h);
	const half3 dotColour = half3(1.0h, 0.35h, 0.55h);
	half3 rgb = mix(base.rgb, lineColour, half(lineAlpha * 0.85));
	rgb = mix(rgb, dotColour, half(dotAlpha));
	outputTexture.write(half4(rgb, base.a), gid);
}
