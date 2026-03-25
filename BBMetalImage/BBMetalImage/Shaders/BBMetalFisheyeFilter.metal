//
//  BBMetalFisheyeFilter.metal
//  BBMetalImage
//
//  Created based on fisheye shader from shadertoy.com/view/ll2GWV
//

#include <metal_stdlib>
#include "BBMetalShaderTypes.h"
using namespace metal;

struct FisheyeUniforms {
    float modifier;
    float distortionMix;
    float borderSoftness;
    float vignetteStrength;
};

kernel void fisheyeKernel(texture2d<half, access::write> outputTexture [[texture(0)]],
                          texture2d<half, access::sample> inputTexture [[texture(1)]],
                          constant FisheyeUniforms &uniforms [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {

    if ((gid.x >= outputTexture.get_width()) || (gid.y >= outputTexture.get_height())) { return; }

    // Pre-calculate constants.
    const float width = float(outputTexture.get_width());
    const float height = float(outputTexture.get_height());
    const float2 invSize = float2(1.0 / width, 1.0 / height);

    // Normalize coordinates to [-1, 1], then correct for aspect so the effect stays circular.
    const float2 uv = float2(float(gid.x), float(gid.y)) * 2.0 * invSize - float2(1.0);
    const float aspect = width / height;
    const float2 uvAspect = float2(uv.x * aspect, uv.y);

    // Compute radial distance in fisheye space.
    const float lensScale = max(0.1, 1.8 - uniforms.modifier);
    const float radius = length(uvAspect) / lensScale;
    const float clampedRadius = min(radius, 0.9999);

    // Calculate fisheye distortion.
    const float z = sqrt(max(0.0, 1.0 - clampedRadius * clampedRadius));
    const float r = atan2(clampedRadius, z) / 3.14159265;
    const float phi = atan2(uvAspect.y, uvAspect.x);

    // Convert back to texture coordinates.
    const float2 distortedUV = float2((r * cos(phi)) / aspect + 0.5, r * sin(phi) + 0.5);
    const float2 safeUV = clamp(distortedUV, float2(0.0), float2(1.0));
    const float2 sourceUV = float2((float(gid.x) + 0.5) * invSize.x, (float(gid.y) + 0.5) * invSize.y);
    const float distortionMix = clamp(uniforms.distortionMix, 0.0, 1.0);
    const float2 finalUV = mix(sourceUV, safeUV, distortionMix);

    constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
    half4 value = inputTexture.sample(quadSampler, finalUV);

    // DSLR-like edge darkening inside the circular lens projection.
    const float vignette = smoothstep(0.35, 1.0, radius);
    const float vignetteMultiplier = 1.0 - uniforms.vignetteStrength * vignette * vignette;
    value.rgb *= half(max(0.0, vignetteMultiplier));

    // Soften the outer border instead of hard clipping.
    const float softness = clamp(uniforms.borderSoftness, 0.01, 0.95);
    const float edgeMask = smoothstep(1.0 - softness, 1.0, radius);
    value.rgb *= half(1.0 - edgeMask);

    outputTexture.write(half4(value.rgb, 1.0), gid);
}
