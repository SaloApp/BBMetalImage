//
//  BBMetalFisheyeFilter.metal
//  BBMetalImage
//
//  Created based on fisheye shader from shadertoy.com/view/ll2GWV
//

#include <metal_stdlib>
#include "BBMetalShaderTypes.h"
using namespace metal;

kernel void fisheyeKernel(texture2d<half, access::write> outputTexture [[texture(0)]],
                          texture2d<half, access::sample> inputTexture [[texture(1)]],
                          constant float *modifier [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    
    if ((gid.x >= outputTexture.get_width()) || (gid.y >= outputTexture.get_height())) { return; }
    
    // Optimize: pre-calculate constants
    const float width = float(outputTexture.get_width());
    const float height = float(outputTexture.get_height());
    const float2 size = float2(width, height);
    const float2 invSize = float2(1.0 / width, 1.0 / height);
    
    // Normalize coordinates to [-1, 1], then correct for aspect so the effect stays circular.
    const float2 uv = float2(float(gid.x), float(gid.y)) * 2.0 * invSize - float2(1.0);
    const float aspect = width / height;
    const float2 uvAspect = float2(uv.x * aspect, uv.y);
    
    // Calculate distance from center
    const float d = length(uvAspect) / (1.8 - *modifier);
    
    // Early exit for pixels outside the fisheye effect
    if (d >= 1.0) {
        outputTexture.write(half4(0.0, 0.0, 0.0, 1.0), gid);
        return;
    }
    
    // Calculate fisheye distortion
    const float z = sqrt(1.0 - d * d);
    const float r = atan2(d, z) / 3.14159;
    const float phi = atan2(uvAspect.y, uvAspect.x);
    
    // Convert back to texture coordinates
    const float2 distortedUV = float2((r * cos(phi)) / aspect + 0.5, r * sin(phi) + 0.5);
    
    constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
    const half4 value = inputTexture.sample(quadSampler, distortedUV);
    outputTexture.write(value, gid);
}
