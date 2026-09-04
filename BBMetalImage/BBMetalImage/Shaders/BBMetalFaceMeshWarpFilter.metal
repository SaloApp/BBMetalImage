#include <metal_stdlib>
using namespace metal;

// One landmark handle: where a point IS, and where it should END UP.
//
// Both live in the frame's UV space. A handle whose `target` equals its `source` is an anchor — it
// pins that part of the image in place. Anchors are not optional decoration: Moving Least Squares
// is a global deformation, so without a ring of them around the face the warp would drag the whole
// frame along with it.
struct FaceMeshControlPoint {
	float2 source;
	float2 target;
	float  weight;
	float  _pad;
};

static inline float2 complexMultiply(float2 a, float2 b) {
	return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Rigid image deformation using Moving Least Squares — Schaefer, McPhail & Warren, SIGGRAPH 2006,
// section 2.3.
//
// WHY THIS AND NOT A SUM OF RADIAL BUMPS. The previous version accumulated a weighted sum of local
// displacement fields. That sum cancels wherever two handles pointing different ways overlap, and
// goes lumpy well before it goes strong — measured, a nominal 38% widening delivered 28%, and
// pushing the numbers up made it bulge rather than deform. MLS instead solves, per pixel, for the
// best RIGID transform that agrees with all the handles, weighted by inverse distance. Rigid means
// no local shear or scale, which is exactly why it survives large deformation: the face moves like
// a face instead of inflating like a balloon.
//
// THE ALGEBRA, CONDENSED. The paper's rigid solution is
//
//     f(v) = |v - p*| · (Σ q̂ᵢ Aᵢ) / |Σ q̂ᵢ Aᵢ| + q*
//
// with Aᵢ built from p̂ᵢ, its perpendicular, and the same pair for (v - p*). Written out, each Aᵢ is
// wᵢ·[[c, s], [-s, c]] where c = p̂ᵢ · d and s = cross(p̂ᵢ, d) — a scaled rotation. In the complex
// plane that whole product collapses to
//
//     Σ wᵢ · q̂ᵢ · conj(p̂ᵢ) · d
//
// and since d is common it factors out, leaving one accumulator S = Σ wᵢ q̂ᵢ conj(p̂ᵢ) and a final
//
//     f(v) = q* + normalize(S) · d          (complex multiplication)
//
// which is three multiply-adds per handle and one normalize per pixel — no matrices, no inverse, no
// per-pixel solve.
//
// DIRECTION. A fragment shader needs the INVERSE map: given an output pixel, where do we sample?
// So the deformation runs from target space to source space — p = target, q = source.
kernel void faceMeshWarpKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant FaceMeshControlPoint *controlPoints [[buffer(0)]],
	constant uint *controlPointCountPtr [[buffer(1)]],
	constant float4 *faceBoundsPtr [[buffer(2)]],
	constant float *strengthPtr [[buffer(3)]],
	constant float *alphaPtr [[buffer(4)]],
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

	const uint count = min(*controlPointCountPtr, 48u);
	const float strength = clamp(*strengthPtr, 0.0, 2.0);
	const float alpha = clamp(*alphaPtr, 0.25, 4.0);

	// Everything outside the padded box costs one sample. The box has to cover the anchor ring, not
	// just the face, because that is where the deformation actually reaches zero.
	const float4 faceBounds = *faceBoundsPtr;
	const bool insideBounds =
		uv.x >= faceBounds.x && uv.x <= faceBounds.z &&
		uv.y >= faceBounds.y && uv.y <= faceBounds.w;

	if (count < 2u || strength <= 0.0001 || !insideBounds) {
		outputTexture.write(inputTexture.sample(quadSampler, uv), gid);
		return;
	}

	// Continuity at the box edge. The anchor ring is supposed to bring the deformation to zero, but
	// it only does so exactly AT the anchors — between them the map still deviates, so cutting hard
	// at the boundary leaves a seam. Measured on a ramp, that seam was a 4.5px backward jump for the
	// contracting presets, visible as a ring around the face. Ramping the deformation down across
	// the outer quarter of the box makes the boundary continuous by construction, for one
	// smoothstep, and it attenuates only the region outside the anchors where the warp is meant to
	// be finished anyway.
	const float2 boxMinimum = faceBounds.xy;
	const float2 boxMaximum = faceBounds.zw;
	const float2 boxHalfExtent = max((boxMaximum - boxMinimum) * 0.5, float2(1e-4));
	const float2 edgeDistance = min(uv - boxMinimum, boxMaximum - uv) / (boxHalfExtent * 0.25);
	const float edgeWindow = smoothstep(
		0.0, 1.0, clamp(min(edgeDistance.x, edgeDistance.y), 0.0, 1.0)
	);
	const float appliedStrength = strength * edgeWindow;
	if (appliedStrength <= 0.0001) {
		outputTexture.write(inputTexture.sample(quadSampler, uv), gid);
		return;
	}

	// --- Weights and weighted centroids ---
	float weightSum = 0.0;
	float2 pStar = float2(0.0);
	float2 qStar = float2(0.0);
	int exactIndex = -1;
	for (uint i = 0u; i < count; ++i) {
		const FaceMeshControlPoint handle = controlPoints[i];
		if (handle.weight <= 0.0) {
			continue;
		}
		const float2 offset = handle.target - uv;
		const float distanceSquared = dot(offset, offset);
		if (distanceSquared < 1e-12) {
			// f(pᵢ) = qᵢ exactly. Taking the limit numerically would divide by zero.
			exactIndex = int(i);
			break;
		}
		const float w = handle.weight / powr(distanceSquared, alpha);
		weightSum += w;
		pStar += w * handle.target;
		qStar += w * handle.source;
	}

	if (exactIndex >= 0) {
		const float2 mapped = controlPoints[exactIndex].source;
		const float2 sourceUV = clamp(uv + (mapped - uv) * appliedStrength, float2(0.0), float2(1.0));
		outputTexture.write(inputTexture.sample(quadSampler, sourceUV), gid);
		return;
	}
	if (weightSum <= 1e-20) {
		outputTexture.write(inputTexture.sample(quadSampler, uv), gid);
		return;
	}

	pStar /= weightSum;
	qStar /= weightSum;
	const float2 d = uv - pStar;

	// --- S = Σ wᵢ · q̂ᵢ · conj(p̂ᵢ) ---
	float2 accumulator = float2(0.0);
	for (uint i = 0u; i < count; ++i) {
		const FaceMeshControlPoint handle = controlPoints[i];
		if (handle.weight <= 0.0) {
			continue;
		}
		const float2 offset = handle.target - uv;
		const float distanceSquared = dot(offset, offset);
		if (distanceSquared < 1e-12) {
			continue;
		}
		const float w = handle.weight / powr(distanceSquared, alpha);
		const float2 pHat = handle.target - pStar;
		const float2 qHat = handle.source - qStar;
		accumulator += w * complexMultiply(qHat, float2(pHat.x, -pHat.y));
	}

	const float accumulatorLength = length(accumulator);
	if (accumulatorLength <= 1e-20) {
		// Degenerate configuration — every handle collapsed onto the centroid.
		outputTexture.write(inputTexture.sample(quadSampler, uv), gid);
		return;
	}

	const float2 mapped = qStar + complexMultiply(accumulator / accumulatorLength, d);
	const float2 sourceUV = clamp(uv + (mapped - uv) * appliedStrength, float2(0.0), float2(1.0));
	outputTexture.write(inputTexture.sample(quadSampler, sourceUV), gid);
}
