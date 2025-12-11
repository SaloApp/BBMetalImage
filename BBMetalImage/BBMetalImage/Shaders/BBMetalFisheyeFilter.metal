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
    
    // Normalize coordinates to [-1, 1]
    const float2 uv = float2(float(gid.x), float(gid.y)) * 2.0 * invSize - float2(1.0);
    
    // Calculate distance from center
    const float d = length(uv) / (2.0 - *modifier);
    
    // Early exit for pixels outside the fisheye effect
    if (d >= 1.0) {
        const float2 texCoord = float2(float(gid.x) * invSize.x, float(gid.y) * invSize.y);
        constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
        const half4 value = inputTexture.sample(quadSampler, texCoord);
        outputTexture.write(value, gid);
        return;
    }
    
    // Calculate fisheye distortion
    const float z = sqrt(1.0 - d * d);
    const float r = atan2(d, z) / 3.14159;
    const float phi = atan2(uv.y, uv.x);
    
    // Convert back to texture coordinates
    const float2 distortedUV = float2(r * cos(phi) + 0.5, r * sin(phi) + 0.5);
    
    constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
    const half4 value = inputTexture.sample(quadSampler, distortedUV);
    outputTexture.write(value, gid);
}
