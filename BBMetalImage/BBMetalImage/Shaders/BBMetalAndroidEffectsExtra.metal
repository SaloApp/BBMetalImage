#include <metal_stdlib>
using namespace metal;

float androidExtraClamp01(float v) {
	return clamp(v, 0.0, 1.0);
}

float2 androidExtraClamp01(float2 v) {
	return clamp(v, 0.0, 1.0);
}

float3 androidExtraClamp01(float3 v) {
	return clamp(v, 0.0, 1.0);
}

float androidExtraRand(float x) {
	return fract(sin(x) * 43758.5453123);
}

float androidExtraRand(float2 st) {
	return fract(sin(dot(st, float2(12.9898, 78.233))) * 43758.5453123);
}

float androidExtraNoise(float2 st) {
	float2 i = floor(st);
	float2 f = fract(st);
	float a = androidExtraRand(i);
	float b = androidExtraRand(i + float2(1.0, 0.0));
	float c = androidExtraRand(i + float2(0.0, 1.0));
	float d = androidExtraRand(i + float2(1.0, 1.0));
	float2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float androidExtraEaseOutSine(float t, float b, float c, float d) {
	constexpr float pi = 3.14159265359;
	return c * sin(t / d * (pi / 2.0)) + b;
}

float2 androidExtraPixelStep(float2 resolution) {
	return float2(1.0 / max(resolution.x, 1.0), 1.0 / max(resolution.y, 1.0));
}

// MARK: - Cathode

float3 androidCathodeSpectrumOffset(float t) {
	float t0 = 3.0 * t - 1.5;
	return clamp(float3(-t0, 1.0 - abs(t0), t0), 0.0, 1.0);
}

float2 androidCathodeBrownConrady(float2 uv, float dist) {
	uv = uv * 2.0 - 1.0;
	float k1 = 0.1 * dist;
	float k2 = -0.025 * dist;
	float r2 = dot(uv, uv);
	uv *= 1.0 + k1 * r2 + k2 * r2 * r2;
	return uv * 0.5 + 0.5;
}

float2 androidCathodeDistort(float2 uv, float t) {
	float2 maxDistort = float2(0.04);
	float2 minDistort = 0.5 * maxDistort;
	float2 dist = mix(minDistort, maxDistort, t);
	return androidCathodeBrownConrady(uv, 75.0 * dist.x);
}

kernel void androidCathodeKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float rnd = androidExtraRand(uv + fract(*timeSec));

	float3 sumCol = float3(0.0);
	float3 sumW = float3(0.0);
	const int numIter = 7;
	const float stepSize = 1.0 / float(numIter - 1);
	float t = rnd * stepSize;

	for (int i = 0; i < numIter; i += 1) {
		float3 w = androidCathodeSpectrumOffset(t);
		sumW += w;
		float2 uvd = androidCathodeDistort(uv, t);
		float3 src = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uvd)).rgb);
		float3 lin = pow(max(src, float3(0.0001)), float3(1.0 / 2.2));
		sumCol += w * lin;
		t += stepSize;
	}

	float3 outCol = sumCol / max(sumW, float3(0.0001));
	outCol = pow(max(outCol, float3(0.0001)), float3(2.2));
	outCol += rnd / 255.0;

	float phase = 0.5 + 0.5 * sin(*timeSec * 0.7);
	float3 phase2 = pow(androidExtraClamp01(outCol), float3(1.6));
	float3 mixed = mix(androidExtraClamp01(outCol), androidExtraClamp01(phase2), phase);
	outputTexture.write(half4(half3(mixed), 1.0), gid);
}

// MARK: - Flash

kernel void androidFlashKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float t = *timeSec;
	float color = 0.35 + 0.35 * sin(t * 10.0);
	float seed = floor(t * 8.0);
	float x = androidExtraRand(seed + 17.0) * 2.0;
	float radius = androidExtraRand(seed + 73.0) * 255.0;

	float2 point = float2(x, 0.0);
	float2 lightDir = point - uv;
	float factor = clamp(10.0 * radius / 255.0 - dot(lightDir, lightDir), 0.0, 1.0);

	float3 base = float3(inputTexture.sample(quadSampler, uv).rgb);
	float3 result = base + factor * float3(color);
	outputTexture.write(half4(half3(androidExtraClamp01(result)), 1.0), gid);
}

// MARK: - Glitch2

float androidGlitch2Hash11(float p) {
	p = fract(p * 0.1031);
	p *= p + 33.33;
	p *= p + p;
	return fract(p);
}

float2 androidGlitch2Hash22(float2 p) {
	float3 p3 = fract(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.xx + p3.yz) * p3.zy);
}

float3 androidGlitch2ShiftColor(texture2d<half, access::sample> inputTexture, float2 uv, float off, sampler quadSampler) {
	float3 clr1 = float3(inputTexture.sample(quadSampler, fract(uv - float2(0.0, off))).rgb);
	float3 clr2 = float3(inputTexture.sample(quadSampler, fract(uv + float2(off, off + off))).rgb);
	float3 clr3 = float3(inputTexture.sample(quadSampler, fract(uv - float2(off, 0.0))).rgb);
	float3 c1 = clr1 * float3(1.000, 0.000, 0.000);
	float3 c2 = clr2 * float3(0.416, 0.353, 0.803);
	float3 c3 = clr3 * float3(0.500, 1.000, 0.831);
	return max(max(float3(0.0), c1), max(c2, c3));
}

float3 androidGlitch2Aberration(texture2d<half, access::sample> inputTexture, float2 uv, float2 off, sampler quadSampler) {
	float r = float(inputTexture.sample(quadSampler, androidExtraClamp01(uv)).r);
	float g = float(inputTexture.sample(quadSampler, androidExtraClamp01(uv - off)).g);
	float b = float(inputTexture.sample(quadSampler, androidExtraClamp01(uv - off - off)).b);
	return float3(r, g, b);
}

float2 androidGlitch2GlitchColor(float2 uv, float lines, float r) {
	float y = floor(uv.y * lines) / lines;
	float offRight = androidExtraRand(float2(r, y));
	float offLeft = androidExtraRand(float2(offRight, y));
	float rn = androidExtraRand(float2(offRight, offLeft));
	float cond = step(offLeft, uv.x) * step(offRight, 1.0 - uv.x) * step(0.95, rn);
	uv.x = mix(uv.x * 0.98, uv.x, cond);
	return uv;
}

float androidGlitch2Stripes(float t, float w, float e) {
	return smoothstep((1.0 - e) / 4.0, (1.0 + e) / 4.0, w * abs(fract(t) - 0.5));
}

float androidGlitch2Spike(float t, float c, float w, float s) {
	return smoothstep(0.0, 1.0, s * w - abs(t - c) * s);
}

float androidGlitch2VerticalAreas(float t) {
	return
		3.0 * androidGlitch2Spike(t, 0.05, 0.095, 40.0) +
		0.5 * androidGlitch2Spike(t, 0.15, 0.035, 40.0) +
		0.2 * androidGlitch2Spike(t, 0.20, 0.032, 40.0) +
		0.7 * androidGlitch2Spike(t, 0.30, 0.025, 40.0) +
		0.7 * androidGlitch2Spike(t, 0.35, 0.037, 40.0) +
		0.5 * androidGlitch2Spike(t, 0.42, 0.050, 40.0) +
		3.0 * androidGlitch2Spike(t, 0.50, 0.050, 40.0) +
		0.1 * androidGlitch2Spike(t, 0.59, 0.055, 40.0) +
		0.2 * androidGlitch2Spike(t, 0.79, 0.090, 10.0) +
		3.0 * androidGlitch2Spike(t, 0.95, 0.100, 40.0);
}

float androidGlitch2WhiteGlitch(float t) {
	t *= 1.5;
	return 0.3333 * step(1.9, fmod(t - 1.4, 3.9)) * step(1.1, fmod(t, 1.9)) * (
		step(0.9, fmod(t, 1.1)) +
		step(0.8, fmod(t, 3.1)) +
		step(0.7, fmod(t, 1.3))
	);
}

kernel void androidGlitch2Kernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float time = *timeSec;

	float lines = (androidGlitch2Stripes(uv.y * 200.0 - uv.x * 13.0, 0.5, 0.9) + 0.3) * androidGlitch2VerticalAreas(uv.x);
	float linesM = androidGlitch2WhiteGlitch(time) * (0.5 + uv.y * uv.y * 0.5);
	lines = min(1.0, lines * linesM);
	uv.x += lines * 0.1;
	uv.y -= lines * 0.05;

	float r = androidExtraRand(float2(floor(time * 7.0) / 7.0));
	uv = androidGlitch2GlitchColor(uv, 30.0, floor(time * 5.0) / 5.0 + r);
	float3 color = androidGlitch2Aberration(inputTexture, uv, float2(0.001, 0.005), quadSampler);

	float4 lbRtCorners;
	float rr = androidExtraRand(float2(floor(time * 3.0), floor(time * 3.0)));
	lbRtCorners.x = androidGlitch2Hash11(rr);
	lbRtCorners.y = androidGlitch2Hash11(lbRtCorners.x);
	lbRtCorners.z = androidGlitch2Hash11(lbRtCorners.y);
	lbRtCorners.w = androidGlitch2Hash11(lbRtCorners.z);

	float cond = step(min(lbRtCorners.x, lbRtCorners.z), uv.x);
	cond *= step(max(lbRtCorners.x, lbRtCorners.z), 1.0 - uv.x);
	cond *= step(min(lbRtCorners.y, lbRtCorners.w), uv.y);
	cond *= step(max(lbRtCorners.y, lbRtCorners.w), 1.0 - uv.y);
	cond *= step(0.4, rr);

	color = max(color, androidGlitch2ShiftColor(inputTexture, uv, 0.04, quadSampler) * cond);
	color += lines;

	outputTexture.write(half4(half3(androidExtraClamp01(color)), 1.0), gid);
}

// MARK: - Glitch3 (single-pass approximation)

kernel void androidGlitch3Kernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float2 px = androidExtraPixelStep(resolution);

	float3 base = float3(inputTexture.sample(quadSampler, uv).rgb);
	float bright = dot(base, float3(0.114, 0.587, 0.299));

	float stripe = floor(uv.x * 110.0 + floor(*timeSec * 8.0));
	float shiftRnd = androidExtraRand(stripe) - 0.5;
	float sortMask = smoothstep(0.25, 0.65, bright);
	float shift = shiftRnd * 0.28 * sortMask;

	float2 sortedUV = float2(uv.x, clamp(uv.y + shift, 0.0, 1.0));
	float3 sorted = float3(inputTexture.sample(quadSampler, sortedUV).rgb);

	float3 c1 = float3(inputTexture.sample(quadSampler, androidExtraClamp01(sortedUV + float2(0.0, 2.0 * px.y))).rgb);
	float3 c2 = float3(inputTexture.sample(quadSampler, androidExtraClamp01(sortedUV - float2(0.0, 2.0 * px.y))).rgb);
	sorted = max(sorted, max(c1, c2));

	float glitchGate = step(0.35, androidExtraRand(float2(floor(*timeSec * 3.0), floor(uv.x * 23.0))));
	float3 color = mix(base, sorted, glitchGate * sortMask);
	outputTexture.write(half4(half3(androidExtraClamp01(color)), 1.0), gid);
}

// MARK: - DSLR Kaleidoscope

float androidDslrBox(float2 st, float2 size) {
	size = 0.5 - size * 0.5;
	float2 uv = step(size, st) * step(size, 1.0 - st);
	return uv.x * uv.y;
}

float androidDslrCircle(float2 uv, float r) {
	float l = length(uv);
	return smoothstep(l - 0.1, l + 0.1, r);
}

float androidDslrRhombus(float2 uv, float r) {
	return 1.0 - smoothstep(1.5, 2.0, abs(uv.x / r) + abs(uv.y / r));
}

kernel void androidDslrKaleidoscopeKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float2 *resolutionIn [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 fallbackResolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 resolution = resolutionIn[0].x > 0.0 && resolutionIn[0].y > 0.0 ? resolutionIn[0] : fallbackResolution;
	float2 uv = (float2(gid) + 0.5) / fallbackResolution;

	float stepCond = step(resolution.x, resolution.y);
	float aspect = mix(resolution.x / max(resolution.y, 1.0), resolution.y / max(resolution.x, 1.0), stepCond);
	float2 dxdy = mix(float2(1.0 / max(aspect, 0.0001), 1.0), float2(1.0, 1.0 / max(aspect, 0.0001)), stepCond);

	float2 uvScaled = uv;
	float isCircle = androidDslrCircle((uvScaled * 2.0 - 1.0) * mix(float2(aspect, 1.0), float2(1.0, aspect), stepCond), 1.0);

	uvScaled *= 4.0;
	uvScaled = fmod(uvScaled, dxdy) + (1.0 - dxdy) * 0.5;

	float2 st = uvScaled * 2.0 - 1.0;
	st *= mix(float2(aspect, 1.0), float2(1.0, aspect), stepCond);

	float isRhombus = androidDslrRhombus(st * 0.9, 0.5);
	float isEdgeBox = androidDslrBox(uvScaled, dxdy * (1.0 - step(0.9999, isRhombus)));

	float3 maybeMix = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uvScaled)).rgb);
	float2 folded = uvScaled - dxdy * (step(0.5, uvScaled) - 0.5);
	float3 col = float3(inputTexture.sample(quadSampler, androidExtraClamp01(folded)).rgb);
	col = mix(col, maybeMix, isRhombus);

	float cond = (isEdgeBox + step(0.9999, isRhombus)) * isCircle;
	outputTexture.write(half4(half3(androidExtraClamp01(col * cond)), 1.0), gid);
}

// MARK: - Kaleidoscope

kernel void androidKaleidoscopeKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 st = (float2(gid) + 0.5) / resolution;
	float2 uv = fract(st * 2.0);
	uv.x = st.x > 0.5 ? (1.0 - uv.x) : uv.x;
	uv.y = st.y < 0.5 ? (1.0 - uv.y) : uv.y;
	float4 color = float4(inputTexture.sample(quadSampler, androidExtraClamp01(uv)));
	outputTexture.write(half4(color), gid);
}

// MARK: - Lumiere

float3 androidLumiereSepia(float3 c) {
	float3 r;
	r.x = dot(c, float3(0.393, 0.769, 0.189));
	r.y = dot(c, float3(0.349, 0.686, 0.168));
	r.z = dot(c, float3(0.272, 0.534, 0.131));
	return r;
}

float3 androidLumiereRgbToYCbCr(float3 c) {
	float3x3 m = float3x3(
		float3(0.2990, -0.1687, 0.5000),
		float3(0.5870, -0.3313, -0.4187),
		float3(0.1140, 0.5000, -0.0813)
	);
	return float3(0.0, 0.5, 0.5) + m * c;
}

float3 androidLumiereYCbCrToRgb(float3 c) {
	float3 v = c - float3(0.0, 0.5, 0.5);
	float3x3 m = float3x3(
		float3(1.0000, 1.0000, 1.0000),
		float3(0.0000, -0.3441, 1.7720),
		float3(1.4020, -0.7141, 0.0000)
	);
	return m * v;
}

float androidLumiereEaseOutQuad(float t) {
	return t * (2.0 - t);
}

float androidLumiereEaseInBounce(float t) {
	return pow(2.0, 6.0 * (t - 1.0)) * abs(sin(t * 3.14 * 3.5));
}

float androidLumiereOffsetY(float timeSec) {
	float fullOffsetY = 0.2;
	float delayTime = 5.0;
	float upTime = 0.5;
	float downTime = 1.0;
	float fullTime = delayTime + upTime + downTime;
	float offsetTime = fmod(timeSec, fullTime);
	float t1 = clamp(offsetTime, 0.0, upTime) / upTime;
	offsetTime -= upTime;
	float t2 = clamp(offsetTime, 0.0, downTime) / downTime;
	return androidLumiereEaseOutQuad(t1) * fullOffsetY + (1.0 - androidLumiereEaseInBounce(t2) * fullOffsetY);
}

kernel void androidLumiereKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv0 = (float2(gid) + 0.5) / resolution;

	float time = *timeSec;
	float2 shake = (float2(androidExtraNoise(float2(0.25 + time * 2.0, 0.25)), androidExtraNoise(float2(0.75 + time * 2.0, 0.75))) - 0.5) * 0.01;
	float2 uv = uv0 + shake;
	uv.y = fract(uv.y - androidLumiereOffsetY(time));
	uv = androidExtraClamp01(uv);

	float3 tc = float3(inputTexture.sample(quadSampler, uv).rgb);
	float3 offTc = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv + float2(0.005, 0.0))).rgb);
	offTc = androidLumiereSepia(offTc);

	float3 colorSepia = float3(androidLumiereSepia(tc).r, offTc.g, offTc.b);
	float3 ycbcr = androidLumiereRgbToYCbCr(colorSepia);
	ycbcr.g += 0.04;
	float3 color = androidLumiereYCbCrToRgb(ycbcr);

	float dirt = step(0.97, androidExtraNoise(uv * 30.0 + time * 20.0));
	float grain = 1.0 + (androidExtraRand(uv + time * 0.01) - 0.2) * 0.15;

	float2 e = uv0 - 0.5;
	e = pow(abs(e), float2(15.0));
	float len = dot(e, float2(1.0));
	float lrb0 = 0.000012;
	float lrb1 = 0.000025;
	float border = len < lrb0 ? 1.0 : (len < lrb1 ? 1.0 - smoothstep(0.0, 1.0, (len - lrb0) / (lrb1 - lrb0)) : 0.0);

	float3 outCol = color * grain + dirt;
	outCol *= border;
	outputTexture.write(half4(half3(androidExtraClamp01(outCol)), 1.0), gid);
}

// MARK: - Pixel Dynamic

kernel void androidPixelDynamicKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	constant float2 *resolutionIn [[buffer(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::nearest, address::clamp_to_edge);
	float2 outRes = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 resolution = resolutionIn[0].x > 0.0 && resolutionIn[0].y > 0.0 ? resolutionIn[0] : outRes;
	float2 uv = (float2(gid) + 0.5) / outRes;

	const float maxPixelSize = 100.0;
	float t = fmod(*timeSec, 3.0);
	t = mix(t / 1.5, 1.0 - (t - 1.5) / 1.5, step(1.5, t));
	float2 pixelSize = clamp(float2(maxPixelSize) * t, float2(1.0), float2(maxPixelSize));
	float2 dxdy = pixelSize / max(resolution, float2(1.0));
	float2 puv = dxdy * floor(uv / dxdy);

	float4 color = float4(inputTexture.sample(quadSampler, androidExtraClamp01(puv)));
	outputTexture.write(half4(color), gid);
}

// MARK: - Pixel Static

kernel void androidPixelStaticKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float2 *resolutionIn [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::nearest, address::clamp_to_edge);
	float2 outRes = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 resolution = resolutionIn[0].x > 0.0 && resolutionIn[0].y > 0.0 ? resolutionIn[0] : outRes;
	float2 uv = (float2(gid) + 0.5) / outRes;
	float2 pixelSize = float2(15.0);
	float2 dxdy = pixelSize / max(resolution, float2(1.0));
	float2 puv = dxdy * floor(uv / dxdy);
	float4 color = float4(inputTexture.sample(quadSampler, androidExtraClamp01(puv)));
	outputTexture.write(half4(color), gid);
}

// MARK: - Polaroid

float4 androidPolaroidGrayscale(float4 color, float factor) {
	float col = dot(color.xyz, float3(0.3, 0.59, 0.11));
	return float4(mix(color.xyz, float3(col), factor), color.w);
}

float4 androidPolaroidVhsColorCorrection(float4 inColor, float add) {
	float avg = androidPolaroidGrayscale(inColor, 0.5).r + add;
	inColor.r *= abs(cos(avg));
	inColor.g *= abs(sin(avg));
	inColor.b *= abs(atan(avg) * sin(avg));
	return inColor + 0.3;
}

kernel void androidPolaroidKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float animDuration = 5.5;
	float time = fmod(*timeSec, animDuration) - 0.5;
	float lerp = 0.0;
	if (time >= 0.0) {
		if (time <= 1.5) {
			lerp = (time / 1.5) * 0.7;
		} else if (time <= 3.5) {
			lerp = 0.7;
		} else if (time <= 5.0) {
			lerp = 0.7 + ((time - 3.5) / 1.5) * (-0.7);
		}
	}

	float4 base = float4(inputTexture.sample(quadSampler, uv));
	float4 developed = androidPolaroidVhsColorCorrection(base, 0.3);
	float4 outColor = mix(base, developed, lerp);
	outputTexture.write(half4(half3(androidExtraClamp01(outColor.rgb)), 1.0), gid);
}

kernel void androidPolaroidPaperKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	texture2d<half, access::sample> paperTexture [[texture(2)]],
	constant float *timeSec [[buffer(0)]],
	constant float *developMixIn [[buffer(1)]],
	constant float *darkThresholdIn [[buffer(2)]],
	constant float *darkSoftnessIn [[buffer(3)]],
	constant float *paperOpacityIn [[buffer(4)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	// Aspect-fill paper texture into output space.
	float outAspect = resolution.x / max(resolution.y, 1.0);
	float paperAspect = float(paperTexture.get_width()) / max(float(paperTexture.get_height()), 1.0);
	float2 paperUV = uv;
	if (paperAspect > outAspect) {
		float normalizedWidth = outAspect / max(paperAspect, 1e-6);
		paperUV.x = (uv.x - 0.5) * normalizedWidth + 0.5;
	} else if (paperAspect < outAspect) {
		float normalizedHeight = paperAspect / max(outAspect, 1e-6);
		paperUV.y = (uv.y - 0.5) * normalizedHeight + 0.5;
	}
	float3 paper = float3(paperTexture.sample(quadSampler, androidExtraClamp01(paperUV)).rgb);

	// Camera scale-down in center (0.9) before aspect mapping.
	const float cameraScale = 0.90;
	float2 cameraCanvasUV = (uv - 0.5) / cameraScale + 0.5;
	bool cameraInBounds = (
		cameraCanvasUV.x >= 0.0 && cameraCanvasUV.x <= 1.0 &&
		cameraCanvasUV.y >= 0.0 && cameraCanvasUV.y <= 1.0
	);

	// Camera aspect-fill in output space.
	float cameraAspect = float(inputTexture.get_width()) / max(float(inputTexture.get_height()), 1.0);
	float2 cameraUV = cameraCanvasUV;
	if (cameraAspect > outAspect) {
		float normalizedWidth = outAspect / max(cameraAspect, 1e-6);
		cameraUV.x = (cameraCanvasUV.x - 0.5) * normalizedWidth + 0.5;
	} else if (cameraAspect < outAspect) {
		float normalizedHeight = cameraAspect / max(outAspect, 1e-6);
		cameraUV.y = (cameraCanvasUV.y - 0.5) * normalizedHeight + 0.5;
	}
	float4 baseCamera = float4(inputTexture.sample(quadSampler, androidExtraClamp01(cameraUV)));
	float4 developedCamera = androidPolaroidVhsColorCorrection(baseCamera, 0.3);
	float developMix = clamp(*developMixIn, 0.0, 1.0);
	float3 cameraColor = mix(baseCamera.rgb, developedCamera.rgb, developMix);

	// Development appears through paper black pixels.
	float paperLuma = dot(paper, float3(0.299, 0.587, 0.114));
	float darkThreshold = clamp(*darkThresholdIn, 0.0, 1.0);
	float darkSoftness = clamp(*darkSoftnessIn, 0.001, 0.5);
	float blackMask = smoothstep(darkThreshold + darkSoftness, darkThreshold - darkSoftness, paperLuma);

	float phase = androidExtraNoise(uv * resolution * 0.018 + float2(*timeSec * 0.09, *timeSec * 0.06));
	float developPhase = mix(0.88, 1.08, phase);
	float reveal = clamp(blackMask * developPhase * max(developMix, 0.15), 0.0, 1.0);
	if (!cameraInBounds) {
		reveal = 0.0;
	}

	float overlayStrength = clamp(*paperOpacityIn, 0.0, 1.0);
	float3 outRgb = mix(paper, cameraColor, reveal * overlayStrength);

	outputTexture.write(half4(half3(androidExtraClamp01(outRgb)), 1.0), gid);
}

// MARK: - Rave

kernel void androidRaveKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float speed = 3.0;
	float time = fmod(*timeSec * speed, 4.0);
	float phase = fract(time);
	float idx = floor(time);

	float3 a = idx < 1.0 ? float3(0.0, 0.0, 1.0) : (idx < 2.0 ? float3(1.0, 0.0, 1.0) : (idx < 3.0 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0)));
	float3 b = idx < 1.0 ? float3(1.0, 0.0, 1.0) : (idx < 2.0 ? float3(1.0, 0.0, 0.0) : (idx < 3.0 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0)));
	float3 c = idx < 1.0 ? float3(0.0, 1.0, 0.0) : (idx < 2.0 ? float3(0.0, 0.0, 1.0) : (idx < 3.0 ? float3(1.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0)));
	float3 d = idx < 1.0 ? float3(1.0, 0.0, 0.0) : (idx < 2.0 ? float3(0.0, 1.0, 0.0) : (idx < 3.0 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 1.0)));

	float3 bl = mix(a, b, phase);
	float3 br = mix(b, d, phase);
	float3 tl = mix(c, a, phase);
	float3 tr = mix(d, c, phase);

	float3 x0 = mix(bl, tr, uv.x);
	float3 x1 = mix(tl, br, uv.x);
	float3 clr = mix(x0, x1, uv.y);
	float3 tex = float3(inputTexture.sample(quadSampler, uv).rgb);
	clr = pow(clr, float3(1.5)) + tex;

	outputTexture.write(half4(half3(androidExtraClamp01(clr)), 1.0), gid);
}

// MARK: - Soul

kernel void androidSoulKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float cycle = 1.2;
	float t = fmod(*timeSec, cycle) / cycle;
	float eased = t < 0.5 ? (2.0 * t * t) : (-1.0 + (4.0 - 2.0 * t) * t);
	float ratio = mix(1.0, 5.0, eased);
	float alpha = 1.0 - eased;

	float rectZ = min(1.0 / ratio, 1.0);
	float rectW = rectZ;
	float2 origin = float2(0.5 - rectZ * 0.5, 0.5 - rectW * 0.5);
	origin = clamp(origin, 0.0, 1.0 - float2(rectZ, rectW));
	float2 modifiedUV = origin + uv * float2(rectZ, rectW);

	float4 base = float4(inputTexture.sample(quadSampler, androidExtraClamp01(uv)));
	float4 modified = float4(inputTexture.sample(quadSampler, androidExtraClamp01(modifiedUV)));
	float4 outColor = mix(base, modified, alpha);
	outputTexture.write(half4(half3(androidExtraClamp01(outColor.rgb)), 1.0), gid);
}

// MARK: - Stars (single-pass approximation)

kernel void androidStarsKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float2 px = androidExtraPixelStep(resolution);

	float3 base = float3(inputTexture.sample(quadSampler, uv).rgb);
	float i = dot(base, float3(0.299, 0.587, 0.114));

	float3 c1 = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv + float2(0.03, 0.0))).rgb);
	float3 c2 = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv - float2(0.03, 0.0))).rgb);
	float3 c3 = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv + float2(0.0, 0.03))).rgb);
	float3 c4 = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv - float2(0.0, 0.03))).rgb);

	float edge = 0.0;
	edge += step(0.5, i - dot(c1, float3(0.299, 0.587, 0.114))) * step(0.5, i - dot(c2, float3(0.299, 0.587, 0.114)));
	edge += step(0.5, i - dot(c3, float3(0.299, 0.587, 0.114))) * step(0.5, i - dot(c4, float3(0.299, 0.587, 0.114)));
	edge = clamp(edge, 0.0, 1.0);

	float streak = 0.0;
	for (int k = 1; k <= 5; k += 1) {
		float w = 1.0 / float(k);
		float2 off = float2(px.x * float(k) * 3.0, px.y * float(k) * 3.0);
		streak += dot(float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv + off)).rgb), float3(0.3333)) * w;
		streak += dot(float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv - off)).rgb), float3(0.3333)) * w;
	}
	streak *= edge * 0.35;
	float3 tint = float3(1.5, 1.0, 0.5);
	float3 outColor = base + tint * streak;
	outputTexture.write(half4(half3(androidExtraClamp01(outColor)), 1.0), gid);
}

// MARK: - TV Foam

float androidTvFoamSnoise(float2 v) {
	float2 i = floor(v);
	float2 f = fract(v);
	float a = androidExtraRand(i);
	float b = androidExtraRand(i + float2(1.0, 0.0));
	float c = androidExtraRand(i + float2(0.0, 1.0));
	float d = androidExtraRand(i + float2(1.0, 1.0));
	float2 u = f * f * (3.0 - 2.0 * f);
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

kernel void androidTvFoamKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float time = *timeSec * 2.0;
	float noise = max(0.0, androidTvFoamSnoise(float2(time, uv.y * 0.3)) - 0.3) * (1.0 / 0.7);
	noise = noise + (androidTvFoamSnoise(float2(time * 10.0, uv.y * 2.4)) - 0.5) * 0.15;

	float xpos = uv.x - noise * noise * 0.25;
	if (xpos < 0.0) { xpos = fract(xpos); }

	float3 color = float3(inputTexture.sample(quadSampler, androidExtraClamp01(float2(xpos, uv.y))).rgb);
	float rnd = androidExtraRand(float2(uv.y * time));
	color = mix(color, float3(rnd), noise * 0.3);

	float g = float(inputTexture.sample(quadSampler, androidExtraClamp01(float2(xpos + noise * 0.01, uv.y))).g);
	float b = float(inputTexture.sample(quadSampler, androidExtraClamp01(float2(xpos - noise * 0.01, uv.y))).b);
	color.g = mix(color.r, g, 0.7);
	color.b = mix(color.r, b, 0.7);

	float foam = fract(sin(fract(*timeSec) / 100.0 * dot(uv, float2(12.9898, 78.233))) * 43758.5453123);
	color *= abs(foam + 0.75);

	outputTexture.write(half4(half3(androidExtraClamp01(color)), 1.0), gid);
}

// MARK: - Twitch

float androidTwitchScale(float timeSec) {
	float t = fmod(timeSec, 1.0);
	if (t <= 0.15) {
		return 1.0 - abs(sin((t / 0.15) * 3.14159265359)) / 2.0;
	}
	if (t <= 0.45) {
		float local = fmod(t, 0.1);
		return 1.0 - abs(sin((local / 0.1) * 3.14159265359)) / 5.0;
	}
	return 1.0;
}

kernel void androidTwitchKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float scale = androidTwitchScale(*timeSec);
	float2 tUv = (uv - 0.5) * scale + 0.5;

	float4 outColor = float4(0.0);
	if (abs(scale - 1.0) > 0.0001) {
		const int samplesCount = 10;
		float precomputedBlurScale = 0.4 * (1.0 - scale) / float(samplesCount - 1);
		for (int i = 0; i < samplesCount; i += 1) {
			float blurScale = 1.0 + precomputedBlurScale * float(i);
			float2 suv = (tUv - 0.5) * blurScale + 0.5;
			outColor += float4(inputTexture.sample(quadSampler, androidExtraClamp01(suv)));
		}
		outColor /= float(samplesCount);
	} else {
		outColor = float4(inputTexture.sample(quadSampler, androidExtraClamp01(tUv)));
	}

	outputTexture.write(half4(half3(androidExtraClamp01(outColor.rgb)), 1.0), gid);
}

// MARK: - DV Cam

constant int kAndroidTextDVCAM[6] = { 68, 86, 32, 67, 65, 77 }; // DV CAM
constant int kAndroidTextPLAY[4] = { 80, 76, 65, 89 }; // PLAY
constant int kAndroidTextSEC60[5] = { 54, 48, 83, 69, 67 }; // 60SEC
constant int kAndroidTextDVIN[5] = { 68, 86, 32, 73, 78 }; // DV IN
constant int kAndroidTextVHS2CAM[7] = { 67, 65, 77, 69, 82, 65, 49 }; // CAMERA1
constant int kAndroidTextSOURCE[6] = { 83, 79, 85, 82, 67, 69 }; // SOURCE

int androidOverlayGlyphRow(int ch, int row) {
	if (row < 0 || row > 6) { return 0; }
	switch (ch) {
	case 32: return 0; // space
	case 48: // 0
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 17; case 3: return 17; case 4: return 17; case 5: return 17; default: return 14; }
	case 49: // 1
		switch (row) { case 0: return 4; case 1: return 12; case 2: return 4; case 3: return 4; case 4: return 4; case 5: return 4; default: return 14; }
	case 50: // 2
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 1; case 3: return 2; case 4: return 4; case 5: return 8; default: return 31; }
	case 51: // 3
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 1; case 3: return 6; case 4: return 1; case 5: return 17; default: return 14; }
	case 52: // 4
		switch (row) { case 0: return 2; case 1: return 6; case 2: return 10; case 3: return 18; case 4: return 31; case 5: return 2; default: return 2; }
	case 53: // 5
		switch (row) { case 0: return 31; case 1: return 16; case 2: return 30; case 3: return 1; case 4: return 1; case 5: return 17; default: return 14; }
	case 54: // 6
		switch (row) { case 0: return 6; case 1: return 8; case 2: return 16; case 3: return 30; case 4: return 17; case 5: return 17; default: return 14; }
	case 55: // 7
		switch (row) { case 0: return 31; case 1: return 1; case 2: return 2; case 3: return 4; case 4: return 8; case 5: return 8; default: return 8; }
	case 56: // 8
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 17; case 3: return 14; case 4: return 17; case 5: return 17; default: return 14; }
	case 57: // 9
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 17; case 3: return 15; case 4: return 1; case 5: return 2; default: return 12; }
	case 58: // :
		switch (row) { case 1: return 4; case 2: return 4; case 4: return 4; case 5: return 4; default: return 0; }
	case 62: // >
		switch (row) { case 0: return 1; case 1: return 2; case 2: return 4; case 3: return 8; case 4: return 4; case 5: return 2; default: return 1; }
	case 65: // A
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 17; case 3: return 31; case 4: return 17; case 5: return 17; default: return 17; }
	case 67: // C
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 16; case 3: return 16; case 4: return 16; case 5: return 17; default: return 14; }
	case 68: // D
		switch (row) { case 0: return 30; case 1: return 17; case 2: return 17; case 3: return 17; case 4: return 17; case 5: return 17; default: return 30; }
	case 69: // E
		switch (row) { case 0: return 31; case 1: return 16; case 2: return 16; case 3: return 30; case 4: return 16; case 5: return 16; default: return 31; }
	case 73: // I
		switch (row) { case 0: return 31; case 1: return 4; case 2: return 4; case 3: return 4; case 4: return 4; case 5: return 4; default: return 31; }
	case 75: // K
		switch (row) { case 0: return 17; case 1: return 18; case 2: return 20; case 3: return 24; case 4: return 20; case 5: return 18; default: return 17; }
	case 76: // L
		switch (row) { case 0: return 16; case 1: return 16; case 2: return 16; case 3: return 16; case 4: return 16; case 5: return 16; default: return 31; }
	case 77: // M
		switch (row) { case 0: return 17; case 1: return 27; case 2: return 21; case 3: return 21; case 4: return 17; case 5: return 17; default: return 17; }
	case 78: // N
		switch (row) { case 0: return 17; case 1: return 25; case 2: return 21; case 3: return 19; case 4: return 17; case 5: return 17; default: return 17; }
	case 79: // O
		switch (row) { case 0: return 14; case 1: return 17; case 2: return 17; case 3: return 17; case 4: return 17; case 5: return 17; default: return 14; }
	case 80: // P
		switch (row) { case 0: return 30; case 1: return 17; case 2: return 17; case 3: return 30; case 4: return 16; case 5: return 16; default: return 16; }
	case 82: // R
		switch (row) { case 0: return 30; case 1: return 17; case 2: return 17; case 3: return 30; case 4: return 20; case 5: return 18; default: return 17; }
	case 83: // S
		switch (row) { case 0: return 15; case 1: return 16; case 2: return 16; case 3: return 14; case 4: return 1; case 5: return 1; default: return 30; }
	case 84: // T
		switch (row) { case 0: return 31; case 1: return 4; case 2: return 4; case 3: return 4; case 4: return 4; case 5: return 4; default: return 4; }
	case 85: // U
		switch (row) { case 0: return 17; case 1: return 17; case 2: return 17; case 3: return 17; case 4: return 17; case 5: return 17; default: return 14; }
	case 86: // V
		switch (row) { case 0: return 17; case 1: return 17; case 2: return 17; case 3: return 17; case 4: return 10; case 5: return 10; default: return 4; }
	case 89: // Y
		switch (row) { case 0: return 17; case 1: return 17; case 2: return 10; case 3: return 4; case 4: return 4; case 5: return 4; default: return 4; }
	default: return 0;
	}
}

float androidOverlayRect(float2 uv, float2 origin, float2 size) {
	float2 p = (uv - origin) / size;
	return step(0.0, p.x) * step(p.x, 1.0) * step(0.0, p.y) * step(p.y, 1.0);
}

float androidOverlayDrawChar(float2 suvTopLeft, float2 origin, float2 charSize, int ch) {
	float2 p = (suvTopLeft - origin) / charSize;
	if (p.x < 0.0 || p.x >= 1.0 || p.y < 0.0 || p.y >= 1.0) { return 0.0; }
	int gx = int(floor(p.x * 5.0));
	int gy = int(floor(p.y * 7.0));
	gx = clamp(gx, 0, 4);
	gy = clamp(gy, 0, 6);
	int rowBits = androidOverlayGlyphRow(ch, gy);
	int bit = (rowBits >> (4 - gx)) & 1;
	return float(bit);
}

float androidOverlayDrawText(
	float2 suvTopLeft,
	float2 origin,
	float2 charSize,
	float spacing,
	constant int *chars,
	int count
) {
	float mask = 0.0;
	for (int i = 0; i < count; i += 1) {
		float2 cOrigin = origin + float2(float(i) * (charSize.x + spacing), 0.0);
		mask = max(mask, androidOverlayDrawChar(suvTopLeft, cOrigin, charSize, chars[i]));
	}
	return mask;
}

float androidOverlayDrawTimerHHMMSSFF(
	float2 suvTopLeft,
	float2 origin,
	float2 charSize,
	float spacing,
	int hh,
	int mm,
	int ss,
	int ff
) {
	int chars[11] = {
		48 + (hh / 10) % 10,
		48 + hh % 10,
		58,
		48 + (mm / 10) % 10,
		48 + mm % 10,
		58,
		48 + (ss / 10) % 10,
		48 + ss % 10,
		58,
		48 + (ff / 10) % 10,
		48 + ff % 10
	};
	float mask = 0.0;
	for (int i = 0; i < 11; i += 1) {
		float2 cOrigin = origin + float2(float(i) * (charSize.x + spacing), 0.0);
		mask = max(mask, androidOverlayDrawChar(suvTopLeft, cOrigin, charSize, chars[i]));
	}
	return mask;
}

float androidOverlayDrawTimerMMSSFF(
	float2 suvTopLeft,
	float2 origin,
	float2 charSize,
	float spacing,
	int mm,
	int ss,
	int ff
) {
	int chars[8] = {
		48 + (mm / 10) % 10,
		48 + mm % 10,
		58,
		48 + (ss / 10) % 10,
		48 + ss % 10,
		58,
		48 + (ff / 10) % 10,
		48 + ff % 10
	};
	float mask = 0.0;
	for (int i = 0; i < 8; i += 1) {
		float2 cOrigin = origin + float2(float(i) * (charSize.x + spacing), 0.0);
		mask = max(mask, androidOverlayDrawChar(suvTopLeft, cOrigin, charSize, chars[i]));
	}
	return mask;
}

float androidDvcamStripes(float t, float e) {
	return smoothstep((1.0 - e) / 4.0, (1.0 + e) / 4.0, abs(fract(t) - 0.5));
}

float2 androidDvcamFuzzyGrid(float2 c, float e) {
	return floor(c) + smoothstep(1.0 - e, 1.0, fract(c));
}

float androidDvcamGlitch1(float t) {
	return step(1.5 - 1.0 / 20.0, fmod(t, 1.5)) + step(2.7 - 1.0 / 20.0, fmod(t, 2.7)) + step(3.9 - 1.0 / 20.0, fmod(t, 3.9));
}

float androidDvcamGlitch2(float t) {
	return step(1.9 - 1.0 / 20.0, fmod(t, 1.9)) + step(2.9 - 1.0 / 20.0, fmod(t, 2.9)) + step(5.9 - 1.0 / 20.0, fmod(t, 5.9));
}

float2 androidDvcamHash22(float2 p) {
	p = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
	return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

kernel void androidDvCamKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 imageSize = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / imageSize;

	float aspect = imageSize.x / max(imageSize.y, 1.0);
	float2 squareMult = imageSize.x > imageSize.y ? float2(1.0, 1.0 / aspect) : float2(aspect, 1.0);
	float2 squareXY = uv * squareMult;

	if (androidDvcamGlitch1(*timeSec) > 0.5) {
		float2 gridCoord = androidDvcamFuzzyGrid(squareXY * 16.0, 0.15) / 16.0;
		gridCoord = gridCoord * (15.0 / 16.0) + 0.5 / 16.0;
		uv += (androidDvcamHash22(gridCoord) * 2.0 - 1.0) * 0.03;
	}

	float3 c = float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv)).rgb);
	float blueShift = float(inputTexture.sample(quadSampler, androidExtraClamp01(uv + float2(-0.014, 0.002))).b);
	c.b = mix(c.b, blueShift, 0.35);

	if (androidDvcamGlitch2(*timeSec) > 0.5) {
		float yvFactor = androidDvcamStripes(uv.y * 5.0, 0.25);
		float2 rg = pow(c.rg, mix(float2(1.0, 1.2), float2(0.9, 1.05), yvFactor));
		c.r = rg.x;
		c.g = rg.y;
	}

	float vline = androidDvcamStripes(squareXY.x * 160.0 + 3.0 * sin(squareXY.y * 12.0), 0.4);
	const float vlineFactor = 0.025;
	c *= vline * vlineFactor + (1.0 - vlineFactor);

	float2 xy = uv * 2.0 - 1.0;
	xy.x *= aspect;
	float r = length(xy) * rsqrt(aspect * aspect + 1.0);
	float rb = clamp((-1.0 / (0.965 - 0.925)) * r + (0.965 / (0.965 - 0.925)), 0.0, 1.0);
	float g = clamp((-1.0 / (0.96 - 0.92)) * r + (0.96 / (0.96 - 0.92)), 0.0, 1.0);
	c *= float3(rb, g, rb);

	int timeMs = int(max(*timeSec, 0.0) * 1000.0);
	int hh = (timeMs / (60 * 60 * 1000)) % 100;
	int mm = (timeMs % (60 * 60 * 1000)) / (60 * 1000);
	int ss = ((timeMs % (60 * 60 * 1000)) % (60 * 1000)) / 1000;
	int ff = (((timeMs % (60 * 60 * 1000)) % (60 * 1000)) % 1000) * 30 / 1000;

	float2 screenUV = (float2(gid) + 0.5) / imageSize;
	float2 suv = screenUV;
	float2 suvFlip = float2(screenUV.x, 1.0 - screenUV.y);
	float2 charSize = float2(0.020, 0.034);
	float spacing = 0.0035;
	float2 sh = float2(1.0 / imageSize.x, 1.0 / imageSize.y);

	float textWidthTimer = 11.0 * charSize.x + 10.0 * spacing;
	float2 timerOrigin = float2(1.0 - textWidthTimer - 0.03, 0.08);
	float2 playOrigin = float2(timerOrigin.x - (4.0 * charSize.x + 3.0 * spacing) - 0.014, timerOrigin.y);
	float2 secOrigin = float2(1.0 - (5.0 * charSize.x + 4.0 * spacing) - 0.03, timerOrigin.y + charSize.y + 0.008);
	float2 dvCamOrigin = float2(0.03, 1.0 - charSize.y * 3.0 - 0.06);
	float2 dvInOrigin = float2(0.03, dvCamOrigin.y + charSize.y + 0.008);

	float textMask = 0.0;
	float shadowMask = 0.0;

	textMask = max(textMask, max(
		androidOverlayDrawTimerHHMMSSFF(suv, timerOrigin, charSize, spacing, hh, mm, ss, ff),
		androidOverlayDrawTimerHHMMSSFF(suvFlip, timerOrigin, charSize, spacing, hh, mm, ss, ff)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawTimerHHMMSSFF(suv + sh, timerOrigin, charSize, spacing, hh, mm, ss, ff),
		androidOverlayDrawTimerHHMMSSFF(suvFlip + sh, timerOrigin, charSize, spacing, hh, mm, ss, ff)
	));

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, playOrigin, charSize, spacing, kAndroidTextPLAY, 4),
		androidOverlayDrawText(suvFlip, playOrigin, charSize, spacing, kAndroidTextPLAY, 4)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, playOrigin, charSize, spacing, kAndroidTextPLAY, 4),
		androidOverlayDrawText(suvFlip + sh, playOrigin, charSize, spacing, kAndroidTextPLAY, 4)
	));

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, secOrigin, charSize, spacing, kAndroidTextSEC60, 5),
		androidOverlayDrawText(suvFlip, secOrigin, charSize, spacing, kAndroidTextSEC60, 5)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, secOrigin, charSize, spacing, kAndroidTextSEC60, 5),
		androidOverlayDrawText(suvFlip + sh, secOrigin, charSize, spacing, kAndroidTextSEC60, 5)
	));

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, dvCamOrigin, charSize, spacing, kAndroidTextDVCAM, 6),
		androidOverlayDrawText(suvFlip, dvCamOrigin, charSize, spacing, kAndroidTextDVCAM, 6)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, dvCamOrigin, charSize, spacing, kAndroidTextDVCAM, 6),
		androidOverlayDrawText(suvFlip + sh, dvCamOrigin, charSize, spacing, kAndroidTextDVCAM, 6)
	));

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, dvInOrigin, charSize, spacing, kAndroidTextDVIN, 5),
		androidOverlayDrawText(suvFlip, dvInOrigin, charSize, spacing, kAndroidTextDVIN, 5)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, dvInOrigin, charSize, spacing, kAndroidTextDVIN, 5),
		androidOverlayDrawText(suvFlip + sh, dvInOrigin, charSize, spacing, kAndroidTextDVIN, 5)
	));

	float bgMask = 0.0;
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, timerOrigin - float2(0.012, 0.008), float2(textWidthTimer + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, timerOrigin - float2(0.012, 0.008), float2(textWidthTimer + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, secOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, secOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, dvCamOrigin - float2(0.012, 0.008), float2(6.0 * charSize.x + 5.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, dvCamOrigin - float2(0.012, 0.008), float2(6.0 * charSize.x + 5.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, dvInOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, dvInOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016))
	));

	c = mix(c, float3(0.02, 0.025, 0.03), clamp(bgMask, 0.0, 1.0) * 0.65);
	c = mix(c, float3(0.0), clamp(shadowMask, 0.0, 1.0) * 0.65);
	c = mix(c, float3(0.95, 0.97, 1.0), clamp(textMask, 0.0, 1.0));
	float debugBand = step(screenUV.y, 0.04);
	c = mix(c, float3(1.0, 0.0, 0.0), debugBand * 0.85);

	outputTexture.write(half4(half3(androidExtraClamp01(c)), 1.0), gid);
}

// MARK: - VHS2

float androidVhs2Grain(float2 uv, float timeSec) {
	float x = (uv.x + 4.0) * (uv.y + 4.0) * (timeSec * 10.0);
	float g = (fmod((fmod(x, 13.0) + 1.0) * (fmod(x, 123.0) + 1.0), 0.01) - 0.005) * 14.0;
	return g;
}

float androidVhs2EaseInOutBounce(float t) {
	if (t < 0.5) {
		return 8.0 * pow(2.0, 8.0 * (t - 1.0)) * abs(sin(t * 3.14 * 7.0));
	}
	return 1.0 - 8.0 * pow(2.0, -8.0 * t) * abs(sin(t * 3.14 * 7.0));
}

float androidVhs2BezierY(float x) {
	float p1 = pow(1.0 - x, 3.0);
	float p2 = 3.0 * x * pow(1.0 - x, 2.0);
	float p3 = 3.0 * x * x * (1.0 - x);
	float p4 = x * x * x;
	float2 p = p1 * float2(0.0, 24.0 / 255.0) +
		p2 * float2(131.0 / 255.0, 148.0 / 255.0) +
		p3 * float2(226.0 / 255.0, 221.0 / 255.0) +
		p4 * float2(1.0, 1.0);
	return p.y;
}

kernel void androidVhs2Kernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	outputTexture.write(half4(1.0, 0.0, 1.0, 1.0), gid);
	return;
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float2 st = uv;
	float time = *timeSec;
	float2 grid = float2(androidExtraRand(time * floor(uv.y * 250.0)) * androidExtraRand(time) * 36.0, 1.0);
	if (grid.x < 0.0) { grid.x = 1.0; }
	st *= grid;
	float2 ipos = floor(st);
	float2 fpos = fract(st);

	float distortion = 0.0;
	float value = androidExtraRand(ipos.yx + grid * 2000.0);
	value *= value;
	float grain = androidVhs2Grain(uv, time);
	if (1.0 - uv.y > 0.1 && 1.0 - uv.y < 0.97) {
		if (value > 0.95) { distortion = 0.6; }
		value = androidExtraRand(grid);
		if (value > 0.05) { distortion = 0.0; }
	} else {
		if (value > 0.92) { distortion = 0.6; }
		grain *= 1.5;
	}

	float2 pix = androidExtraPixelStep(resolution);
	float3 tex =
		5.0 * float3(inputTexture.sample(quadSampler, uv).rgb) -
		float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv + float2(pix.x, 0.0))).rgb) -
		float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv - float2(pix.x, 0.0))).rgb) -
		float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv + float2(0.0, pix.y))).rgb) -
		float3(inputTexture.sample(quadSampler, androidExtraClamp01(uv - float2(0.0, pix.y))).rgb);

	tex.r = androidVhs2BezierY(tex.r);
	tex.g = androidVhs2BezierY(tex.g);
	tex.b = androidVhs2BezierY(tex.b);

	distortion *= androidVhs2EaseInOutBounce(1.0 - fpos.x);
	float3 color = tex + float3(distortion) + grain * 0.4;

	int timeMs = int(max(*timeSec, 0.0) * 1000.0);
	int mm = (timeMs % (60 * 60 * 1000)) / (60 * 1000);
	int ss = ((timeMs % (60 * 60 * 1000)) % (60 * 1000)) / 1000;
	int ff = (((timeMs % (60 * 60 * 1000)) % (60 * 1000)) % 1000) * 30 / 1000;

	float2 screenUV = (float2(gid) + 0.5) / resolution;
	float2 suv = screenUV;
	float2 suvFlip = float2(screenUV.x, 1.0 - screenUV.y);
	float2 charSize = float2(0.019, 0.032);
	float spacing = 0.0032;
	float2 sh = float2(1.0 / resolution.x, 1.0 / resolution.y);

	float2 camOrigin = float2(0.03, 0.09);
	float2 playOrigin = float2(0.03, camOrigin.y + charSize.y + 0.008);
	float2 timerOrigin = float2(0.03, playOrigin.y + charSize.y + 0.008);
	float2 sourceOrigin = float2(1.0 - (6.0 * charSize.x + 5.0 * spacing) - 0.03, camOrigin.y);

	float textMask = 0.0;
	float shadowMask = 0.0;

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, camOrigin, charSize, spacing, kAndroidTextVHS2CAM, 7),
		androidOverlayDrawText(suvFlip, camOrigin, charSize, spacing, kAndroidTextVHS2CAM, 7)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, camOrigin, charSize, spacing, kAndroidTextVHS2CAM, 7),
		androidOverlayDrawText(suvFlip + sh, camOrigin, charSize, spacing, kAndroidTextVHS2CAM, 7)
	));

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, playOrigin, charSize, spacing, kAndroidTextPLAY, 4),
		androidOverlayDrawText(suvFlip, playOrigin, charSize, spacing, kAndroidTextPLAY, 4)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, playOrigin, charSize, spacing, kAndroidTextPLAY, 4),
		androidOverlayDrawText(suvFlip + sh, playOrigin, charSize, spacing, kAndroidTextPLAY, 4)
	));

	textMask = max(textMask, max(
		androidOverlayDrawTimerMMSSFF(suv, timerOrigin, charSize, spacing, mm, ss, ff),
		androidOverlayDrawTimerMMSSFF(suvFlip, timerOrigin, charSize, spacing, mm, ss, ff)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawTimerMMSSFF(suv + sh, timerOrigin, charSize, spacing, mm, ss, ff),
		androidOverlayDrawTimerMMSSFF(suvFlip + sh, timerOrigin, charSize, spacing, mm, ss, ff)
	));

	textMask = max(textMask, max(
		androidOverlayDrawText(suv, sourceOrigin, charSize, spacing, kAndroidTextSOURCE, 6),
		androidOverlayDrawText(suvFlip, sourceOrigin, charSize, spacing, kAndroidTextSOURCE, 6)
	));
	shadowMask = max(shadowMask, max(
		androidOverlayDrawText(suv + sh, sourceOrigin, charSize, spacing, kAndroidTextSOURCE, 6),
		androidOverlayDrawText(suvFlip + sh, sourceOrigin, charSize, spacing, kAndroidTextSOURCE, 6)
	));

	float bgMask = 0.0;
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, camOrigin - float2(0.012, 0.008), float2(7.0 * charSize.x + 6.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, camOrigin - float2(0.012, 0.008), float2(7.0 * charSize.x + 6.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, timerOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, timerOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidOverlayRect(suv, sourceOrigin - float2(0.012, 0.008), float2(6.0 * charSize.x + 5.0 * spacing + 0.024, charSize.y + 0.016)),
		androidOverlayRect(suvFlip, sourceOrigin - float2(0.012, 0.008), float2(6.0 * charSize.x + 5.0 * spacing + 0.024, charSize.y + 0.016))
	));

	color = mix(color, float3(0.02, 0.025, 0.03), clamp(bgMask, 0.0, 1.0) * 0.65);
	color = mix(color, float3(0.0), clamp(shadowMask, 0.0, 1.0) * 0.65);
	color = mix(color, float3(0.95, 0.97, 1.0), clamp(textMask, 0.0, 1.0));
	float debugBand = step(screenUV.y, 0.04);
	color = mix(color, float3(1.0, 0.0, 0.0), debugBand * 0.85);

	outputTexture.write(half4(half3(androidExtraClamp01(color)), 1.0), gid);
}

// MARK: - Zoom

kernel void androidZoomKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	const float phaseDuration = 0.7;
	const float delayDuration = 0.5;
	const float startValue1 = 1.0;
	const float changeValue1 = -0.3;
	const float startValue2 = startValue1 + changeValue1;
	const float changeValue2 = -changeValue1;
	const float startValue3 = startValue2 + changeValue2;
	const float changeValue3 = 1.25 * changeValue1;
	const float startValue4 = startValue3 + changeValue3;
	const float changeValue4 = -changeValue3;

	float time = fmod(*timeSec, 7.0 * phaseDuration + delayDuration) - delayDuration;
	if (time >= 0.0) {
		time = fmod(time, 7.0 * phaseDuration);
		float2 p = uv - 0.5;
		if (time < 1.0 * phaseDuration) {
			p *= androidExtraEaseOutSine(fmod(time, phaseDuration), startValue1, changeValue1, phaseDuration);
		} else if (time < 2.0 * phaseDuration) {
			p *= androidExtraEaseOutSine(phaseDuration, startValue1, changeValue1, phaseDuration);
		} else if (time < 3.0 * phaseDuration) {
			p *= androidExtraEaseOutSine(fmod(time, phaseDuration), startValue2, changeValue2, phaseDuration);
		} else if (time < 4.0 * phaseDuration) {
			p *= androidExtraEaseOutSine(phaseDuration, startValue2, changeValue2, phaseDuration);
		} else if (time < 5.0 * phaseDuration) {
			p *= androidExtraEaseOutSine(fmod(time, phaseDuration), startValue3, changeValue3, phaseDuration);
		} else if (time < 6.0 * phaseDuration) {
			p *= androidExtraEaseOutSine(phaseDuration, startValue3, changeValue3, phaseDuration);
		} else {
			p *= androidExtraEaseOutSine(fmod(time, phaseDuration), startValue4, changeValue4, phaseDuration);
		}
		uv = p + 0.5;
	}

	float4 color = float4(inputTexture.sample(quadSampler, androidExtraClamp01(uv)));
	outputTexture.write(half4(color), gid);
}

// MARK: - Zoom2

kernel void androidZoom2Kernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float2 p = uv - 0.5;
	float d = length(p);
	// Keep ZM2 at maximum intensity without time-based animation.
	const float k = -1.0;
	d = d * (1.0 + k * d * d);
	float2 n = length(p) > 0.0 ? normalize(p) : float2(0.0, 0.0);
	float2 sampleUV = 0.5 + n * d;

	float4 color = float4(inputTexture.sample(quadSampler, androidExtraClamp01(sampleUV)));
	outputTexture.write(half4(color), gid);
}
