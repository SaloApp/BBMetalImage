#include <metal_stdlib>
using namespace metal;

// Triangle-mesh face warp.
//
// The deformation is carried by GEOMETRY rather than by a per-pixel solve. Each vertex is drawn at
// its deformed position and samples the source at its original one, so the rasteriser does the
// interpolation and the whole warp costs one texture fetch per pixel. That is the difference
// between this and the MLS kernel it replaces: the same deformation, but the expensive part moves
// from ~200k pixels a frame to 500 vertices a detection.
//
// It is also what makes large deformation hold together. A piecewise-affine map over a dense mesh
// has no lumps to develop — it stays well behaved until vertices actually cross, which is a
// condition you can see rather than a gradual smearing.
struct FaceMeshVertexIn {
	// Where the vertex ends up, in frame UV.
	float2 deformed;
	// Where it samples from, in frame UV.
	float2 source;
};

struct FaceMeshVertexOut {
	float4 position [[position]];
	float2 sourceUV;
};

vertex FaceMeshVertexOut faceMeshRenderVertex(
	constant FaceMeshVertexIn *vertices [[buffer(0)]],
	uint vertexID [[vertex_id]]
) {
	const FaceMeshVertexIn input = vertices[vertexID];
	FaceMeshVertexOut output;
	// UV (origin top-left, y down) into clip space (origin centre, y up).
	output.position = float4(
		input.deformed.x * 2.0 - 1.0,
		1.0 - input.deformed.y * 2.0,
		0.0,
		1.0
	);
	output.sourceUV = input.source;
	return output;
}

fragment half4 faceMeshRenderFragment(
	FaceMeshVertexOut input [[stage_in]],
	texture2d<half, access::sample> sourceTexture [[texture(0)]]
) {
	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
	return sourceTexture.sample(quadSampler, clamp(input.sourceUV, float2(0.0), float2(1.0)));
}

// Solid colour, for stroking the tessellation as lines over the frame.
fragment half4 faceMeshWireframeFragment(
	FaceMeshVertexOut input [[stage_in]],
	constant float4 *colourPtr [[buffer(0)]]
) {
	(void)input;
	const float4 colour = *colourPtr;
	return half4(half3(colour.rgb), half(colour.a));
}
