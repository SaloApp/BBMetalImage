#include <metal_stdlib>
using namespace metal;

kernel void yopeVideoWriterKernel(
    texture2d<float, access::write> dst [[texture(0)]],
    texture2d<float, access::sample> src [[texture(1)]],
    constant bool &flipH [[buffer(0)]],
    constant bool &flipV [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dstW = dst.get_width();
    uint dstH = dst.get_height();
    if (gid.x >= dstW || gid.y >= dstH) return;

    float2 uv = (float2(gid) + 0.5) / float2(dstW, dstH);

    float srcW = src.get_width();
    float srcH = src.get_height();
    float2 suv = uv * float2(srcW, srcH);

    if (flipH) suv.x = (srcW - 1.0) - suv.x;
    if (flipV) suv.y = (srcH - 1.0) - suv.y;

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float2 suvNorm = (suv + 0.5) / float2(srcW, srcH);

    float4 c = src.sample(s, suvNorm);
    dst.write(c, gid);
}
