#include <metal_stdlib>
using namespace metal;

kernel void yopeVideoWriterKernel(
    texture2d<float, access::write> dst [[texture(0)]],
    texture2d<float, access::sample> src [[texture(1)]],
    constant bool &flipH [[buffer(0)]],
    constant bool &flipV [[buffer(1)]],
    constant int &rotation [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint dstW = dst.get_width();
    uint dstH = dst.get_height();
    if (gid.x >= dstW || gid.y >= dstH) return;

    float2 uv = (float2(gid) + 0.5) / float2(dstW, dstH);
    float2 suvNorm = uv;
    if (rotation == 1) {
        suvNorm = float2(uv.y, 1.0 - uv.x);
    } else if (rotation == 2) {
        suvNorm = float2(1.0 - uv.y, uv.x);
    }

    if (flipH) suvNorm.x = 1.0 - suvNorm.x;
    if (flipV) suvNorm.y = 1.0 - suvNorm.y;

    constexpr sampler s(address::clamp_to_edge, filter::linear);

    float4 c = src.sample(s, suvNorm);
    dst.write(c, gid);
}
