//
//  BBMetalVHSFilter.metal
//  BBMetalImage
//
//  Created based on VHS shader from shadertoy.com/view/XlsczN
//

#include <metal_stdlib>
#include "BBMetalShaderTypes.h"
using namespace metal;

// RGB to YIQ color space conversion
half3 rgb2yiq(half3 c) {
    return half3(
        (0.2989 * c.x + 0.5959 * c.y + 0.2115 * c.z),
        (0.5870 * c.x - 0.2744 * c.y - 0.5229 * c.z),
        (0.1140 * c.x - 0.3216 * c.y + 0.3114 * c.z)
    );
}

// YIQ to RGB color space conversion
half3 yiq2rgb(half3 c) {
    return half3(
        (1.0 * c.x + 1.0 * c.y + 1.0 * c.z),
        (0.956 * c.x - 0.2720 * c.y - 1.1060 * c.z),
        (0.6210 * c.x - 0.6474 * c.y + 1.7046 * c.z)
    );
}

// Generate circle offset for blur sampling
float2 circle(float start, float points, float point) {
    float rad = (3.141592 * 2.0 * (1.0 / points)) * (point + start);
    return float2(-(0.3 + rad), cos(rad));
}

// Hash function for procedural noise (mimics texture sampling)
float hash(float2 p) {
    return fract(sin(dot(p, float2(127.1, 311.7) )) * 43758.5453);
}

// Generate noise value that mimics iChannel1 texture sampling
// Uses screen coordinates for first sample (like texture(iChannel1, fragCoord))
// In Shadertoy, noise textures are typically smooth/blurred noise, not high-frequency
float noiseChannel1(float2 fragCoord, float time) {
    // Sample at screen coordinates with time-based offset for animation
    // Use very coarse scale to create smooth gradients instead of fine lines
    // Normalize by resolution to make it consistent
    float2 p = fragCoord * 0.0001 + float2(time * 0.2, 0.0);
    return hash(p);
}

// Generate noise for second iChannel1 sample
float noiseChannel1UV(float2 uv, float time) {
    // Similar to texture(iChannel1, vec2(0.01+(uv.y*32.0)/32.0, 1.0)).r
    // Note: (uv.y*32.0)/32.0 simplifies to uv.y, but keeping original formula
    float2 p = float2(0.01 + (uv.y * 32.0) / 32.0, 1.0) + float2(time * 0.1, 0.0);
    return hash(p * 2.0);
}

// Generate noise for iChannel2 (time-based, animated)
float noiseChannel2(float time) {
    // Similar to texture(iChannel2, vec2(mod(iTime*10.0, mod(iTime*10.0, 256.0)*(1.0/256.0)), 0.0)).r
    // The nested mod: mod(iTime*10.0, mod(iTime*10.0, 256.0)*(1.0/256.0))
    float t = time * 10.0;
    float innerMod = mod(t, 256.0) * (1.0 / 256.0);
    // Handle edge case where innerMod might be 0
    float outerMod = innerMod > 0.001 ? mod(t, innerMod) : mod(t, 1.0);
    float2 p = float2(outerMod, time * 0.2);
    return hash(p * 5.0);
}

// Blur function with circular sampling pattern (matches original exactly)
half3 blur(texture2d<half, access::sample> inputTexture, float2 uv, float f, float d, float time) {
    constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);
    
    // t = 0.0 (as in original)
    float t = 0.0;
    float b = 1.0;
    float2 pixelOffset = float2(d + 0.0005 * t, 0.0);
    
    float start = 2.0 / 14.0;
    float2 scale = 0.66 * 4.0 * 2.0 * pixelOffset;
    
    half3 n0 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 0.0) * scale).rgb;
    half3 n1 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 1.0) * scale).rgb;
    half3 n2 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 2.0) * scale).rgb;
    half3 n3 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 3.0) * scale).rgb;
    half3 n4 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 4.0) * scale).rgb;
    half3 n5 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 5.0) * scale).rgb;
    half3 n6 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 6.0) * scale).rgb;
    half3 n7 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 7.0) * scale).rgb;
    half3 n8 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 8.0) * scale).rgb;
    half3 n9 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 9.0) * scale).rgb;
    half3 n10 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 10.0) * scale).rgb;
    half3 n11 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 11.0) * scale).rgb;
    half3 n12 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 12.0) * scale).rgb;
    half3 n13 = inputTexture.sample(quadSampler, uv + circle(start, 14.0, 13.0) * scale).rgb;
    half3 n14 = inputTexture.sample(quadSampler, uv).rgb;
    
    half w = 1.0 / 15.0;
    half3 clr = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n13 + n14) * w;
    
    return clr * b;
}

kernel void vhsKernel(texture2d<half, access::write> outputTexture [[texture(0)]],
                     texture2d<half, access::sample> inputTexture [[texture(1)]],
                     constant float *strength [[buffer(0)]],
                     constant float *time [[buffer(1)]],
                     uint2 gid [[thread_position_in_grid]]) {
    
    if ((gid.x >= outputTexture.get_width()) || (gid.y >= outputTexture.get_height())) { return; }
    
    const float width = float(outputTexture.get_width());
    const float height = float(outputTexture.get_height());
    const float2 resolution = float2(width, height);
    const float2 invSize = float2(1.0 / width, 1.0 / height);
    
    // fragCoord equivalent (screen coordinates)
    float2 fragCoord = float2(float(gid.x), float(gid.y));
    // UV coordinates
    float2 uv = fragCoord * invSize;
    
    // Use strength parameter instead of mouse position (mapped to 0-50 range like iMouse.x)
    float d = 0.1 * (*strength * 50.0) / 50.0;
    
    // Sample noise channel 1 at screen coordinates (like texture(iChannel1, fragCoord).r)
    // Scale noise down by ~10x to reduce intensity
    float s = noiseChannel1(fragCoord, *time) * 0.1;
    
    // Calculate edge distortion
    float e = min(0.30, pow(max(0.0, cos(uv.y * 4.0 + 0.3) - 0.75) * (s + 0.5) * 1.0, 3.0)) * 25.0;
    // Subtract second noise sample (like texture(iChannel1, vec2(0.01+(uv.y*32.0)/32.0, 1.0)).r)
    s -= pow(noiseChannel1UV(uv, *time) * 0.1, 1.0);
    uv.x += e * abs(s * 3.0);
    
    // Add random distortion from channel 2 - scale down significantly
    float r = noiseChannel2(*time) * 0.03 * (2.0 * s);
    uv.x += abs(r * pow(min(0.003, (uv.y - 0.15)) * 6.0, 2.0));
    
    d = 0.051 + abs(sin(s / 4.0));
    float c = max(0.0001, 0.002 * d);
    float2 uvo = uv;
    
    // First blur pass for Y channel
    half3 colorY = blur(inputTexture, uv, 0.0, c + c * uv.x, *time);
    half y = rgb2yiq(colorY).r;
    
    // Second blur pass for I channel
    uv.x += 0.01 * d;
    c *= 6.0;
    half3 colorI = blur(inputTexture, uv, 0.333, c, *time);
    half i = rgb2yiq(colorI).g;
    
    // Third blur pass for Q channel
    uv.x += 0.005 * d;
    c *= 2.50;
    half3 colorQ = blur(inputTexture, uv, 0.666, c, *time);
    half q = rgb2yiq(colorQ).b;
    
    // Convert back to RGB and apply final effects
    half3 finalColor = yiq2rgb(half3(y, i, q));
    // Scale down noise impact in final color by ~10x
    finalColor -= pow(s * 0.1 + e * 2.0, 3.0) * 0.1;
    finalColor *= smoothstep(1.0, 0.999, uv.x - 0.1);
    
    // Clamp to valid range
    finalColor = clamp(finalColor, 0.0, 1.0);
    
    outputTexture.write(half4(half3(finalColor), 1.0), gid);
}

