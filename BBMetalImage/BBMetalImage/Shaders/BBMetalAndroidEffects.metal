#include <metal_stdlib>
using namespace metal;

// MARK: - Acid

float androidAcidVerticalBar(float pos, float uvY, float offset) {
	float edge0 = pos - 0.05;
	float edge1 = pos + 0.05;
	float x = smoothstep(edge0, pos, uvY) * offset;
	x -= smoothstep(pos, edge1, uvY) * offset;
	return x;
}

float2 androidAcidWaveUV(float2 uv, float timeSec) {
	for (int idx = 0; idx < 5; idx += 1) {
		float i = 0.1313 * float(idx);
		float d = fmod(-timeSec * i, 1.5);
		float o = sin(20.0 * i * 0.01);
		o *= 0.25; // offsetIntensity * 5.0
		uv.x += androidAcidVerticalBar(d, uv.y, o);
	}
	return uv;
}

float4 androidAcidColorShift(
	texture2d<half, access::sample> inputTexture,
	float2 uv,
	float timeSec,
	sampler quadSampler
) {
	uv = androidAcidWaveUV(uv, timeSec);
	const float offc = 0.1;
	float3x3 cmat = float3x3(
		float3(1.0, offc, -offc),
		float3(-offc, 1.0, offc),
		float3(offc, -offc, 1.0)
	);

	float y = fract(uv.y + timeSec / 2.0);
	float tear = min(y * 0.05, 1.5 * (1.0 - y));
	float2 sampleUV = clamp(uv, 0.0, 1.0);
	float g = float(inputTexture.sample(quadSampler, sampleUV).g);

	sampleUV = clamp(float2(uv.x, uv.y - 0.14 * (tear + 0.01)), 0.0, 1.0);
	float r = float(inputTexture.sample(quadSampler, sampleUV).r);

	sampleUV = clamp(float2(uv.x, uv.y + 0.26 * (tear + 0.01)), 0.0, 1.0);
	float b = float(inputTexture.sample(quadSampler, sampleUV).b);

	float3 clr = float3(r, g, b);
	float3 result = cmat * clr * 1.1;
	float3 tonemapBase = max((result * 255.0) / (result * 255.0 + 1.0), float3(0.0001));
	result /= tonemapBase;
	return float4(clamp(result, 0.0, 1.0), 1.0);
}

kernel void androidAcidKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float4 color = androidAcidColorShift(inputTexture, uv, *timeSec, quadSampler);
	outputTexture.write(half4(color), gid);
}

// MARK: - Glitch

float androidGlitchRand(float n) {
	return fract(sin(n) * 43758.5453123);
}

float androidGlitchRand(float2 co) {
	return fract(sin(dot(co, float2(12.9898, 78.233))) * 43758.5453);
}

float2 androidGlitchHash(float2 x) {
	const float2 k = float2(0.3183099, 0.3678794);
	x = x * k + k.yx;
	return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

float androidGlitchPerlinNoise(float2 p) {
	float2 i = floor(p);
	float2 f = fract(p);
	float2 u = f * f * (3.0 - 2.0 * f);
	float a = dot(androidGlitchHash(i + float2(0.0, 0.0)), f - float2(0.0, 0.0));
	float b = dot(androidGlitchHash(i + float2(1.0, 0.0)), f - float2(1.0, 0.0));
	float c = dot(androidGlitchHash(i + float2(0.0, 1.0)), f - float2(0.0, 1.0));
	float d = dot(androidGlitchHash(i + float2(1.0, 1.0)), f - float2(1.0, 1.0));
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float androidGlitchNoise(float p) {
	float fl = floor(p);
	float fc = fract(p);
	return mix(androidGlitchRand(fl), androidGlitchRand(fl + 1.0), fc);
}

float androidGlitchBlockyNoise(float2 uv, float threshold, float scale, float seed, float timeSec) {
	float scroll = floor(timeSec + sin(11.0 * timeSec) + sin(timeSec)) * 0.77;
	float2 noiseUV = uv.yy / scale + scroll;
	float noise2 = androidGlitchPerlinNoise(noiseUV);
	float id = floor(noise2 * 20.0);
	id = androidGlitchNoise(id + seed) - 0.5;
	if (abs(id) > threshold) {
		id = 0.0;
	}
	return id;
}

float3 androidGlitchYandexColor(float timeSec) {
	const float3 colorA = float3(0.353, 0.910, 0.000);
	const float3 colorB = float3(1.000, 0.310, 0.137);
	float time = fract(timeSec);
	if (fract(androidGlitchRand(float2(time, pow(time, colorA.g)))) < 0.4) {
		return colorA;
	}
	return colorB;
}

float4 androidGlitchMain(
	texture2d<half, access::sample> inputTexture,
	float2 uv,
	float timeSec,
	sampler quadSampler
) {
	float rgbIntensity = 0.1 + 0.1 * sin(timeSec * 7.7);
	float displaceIntensity = 0.2 + 0.3 * pow(sin(timeSec * 1.8), 5.0);
	float interlaceIntensity = 0.045;

	float displace = androidGlitchBlockyNoise(uv + float2(uv.y, 0.0), displaceIntensity, 0.5, 66.6, timeSec);
	displace *= androidGlitchBlockyNoise(uv.yx + float2(0.0, uv.x), displaceIntensity, 111.0, 13.7, timeSec);

	float2 displacedUV = uv;
	displacedUV.x += displace;

	float2 offs = 0.1 * float2(
		androidGlitchBlockyNoise(displacedUV + float2(displacedUV.y, 0.0), rgbIntensity, 65.0, 341.0, timeSec),
		0.0
	);

	float colr = float(inputTexture.sample(quadSampler, clamp(displacedUV - offs, 0.0, 1.0)).r);
	float colg = float(inputTexture.sample(quadSampler, clamp(displacedUV, 0.0, 1.0)).g);
	float colb = float(inputTexture.sample(quadSampler, clamp(displacedUV + offs, 0.0, 1.0)).b);

	float3 mask = androidGlitchYandexColor(timeSec);
	float maskNoise = androidGlitchBlockyNoise(displacedUV, interlaceIntensity, 90.0, timeSec, timeSec) * max(displace, offs.x);
	maskNoise = 1.0 - maskNoise;

	if (maskNoise == 1.0 || displace < 0.01) {
		return float4(colr, colg, colb, 1.0);
	}
	return float4(mask, 1.0);
}

kernel void androidGlitchKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float4 color = androidGlitchMain(inputTexture, uv, *timeSec, quadSampler);
	outputTexture.write(half4(clamp(color, 0.0, 1.0)), gid);
}

// MARK: - HeatMap

float androidHeatMapBlendOverlay(float base, float blend) {
	return base < 0.5 ? (2.0 * base * blend) : (1.0 - 2.0 * (1.0 - base) * (1.0 - blend));
}

float3 androidHeatMapBlendOverlay(float3 base, float3 blend) {
	return float3(
		androidHeatMapBlendOverlay(base.r, blend.r),
		androidHeatMapBlendOverlay(base.g, blend.g),
		androidHeatMapBlendOverlay(base.b, blend.b)
	);
}

float3 androidHeatMapBlendLighter(float3 a, float3 b) {
	return a.r < b.r ? b : a;
}

float4 androidHeatMapTexSub(float y) {
	return float4(1.0 - y, 0.0, 0.0, 1.0);
}

float4 androidHeatMapTexLight() {
	return float4(0.3765, 0.0, 1.0, 1.0);
}

kernel void androidHeatMapKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float4 sampled = float4(inputTexture.sample(quadSampler, uv));
	float4 bgColor = sampled.bgra;

	float4 texColor2 = androidHeatMapTexLight();
	float4 texColor3 = androidHeatMapTexSub(pow(uv.y, 3.0));

	float3 color2 = androidHeatMapBlendLighter(bgColor.rgb, texColor2.rgb);
	float3 color3 = androidHeatMapBlendOverlay(color2, texColor3.rgb);

	outputTexture.write(half4(half3(color3.b, color3.g, color3.r), half(bgColor.a)), gid);
}

// MARK: - VHS

float androidVhsGrain(float2 uv, float timeSec) {
	float x = (uv.x + 4.0) * (uv.y + 4.0) * (timeSec * 10.0);
	float g = (fmod((fmod(x, 13.0) + 1.0) * (fmod(x, 123.0) + 1.0), 0.01) - 0.005) * 14.0;
	return g;
}

float androidVhsRandom(float x) {
	return fract(sin(x) * 10000.0);
}

float androidVhsRandom(float2 st) {
	return fract(sin(dot(st, float2(12.9898, 78.233))) * 43758.5453123);
}

float androidVhsEaseInOutBounce(float t) {
	if (t < 0.5) {
		return 8.0 * pow(2.0, 8.0 * (t - 1.0)) * abs(sin(t * 3.14 * 7.0));
	}
	return 1.0 - 8.0 * pow(2.0, -8.0 * t) * abs(sin(t * 3.14 * 7.0));
}

float androidVhsGetYPosCubicBezier(float x) {
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

kernel void androidVhsKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;

	float2 st = uv;
	float time = *timeSec;
	float2 grid = float2(androidVhsRandom(time * floor(uv.y * 350.0)) * androidVhsRandom(time) * 36.0, 1.0);
	if (grid.x < 0.0) {
		grid.x = 1.0;
	}
	st *= grid;

	float2 ipos = floor(st);
	float2 fpos = fract(st);
	float distortion = 0.0;

	float value = androidVhsRandom(float2(ipos.y, ipos.x) + grid * 2000.0);
	value *= value;

	float grainValue = androidVhsGrain(uv, time);
	if (1.0 - uv.y > 0.1 && 1.0 - uv.y < 0.97) {
		if (value > 0.95) {
			distortion = 0.6;
		}
		value = androidVhsRandom(grid);
		if (value > 0.05) {
			distortion = 0.0;
		}
	} else {
		if (value > 0.92) {
			distortion = 0.6;
		}
		grainValue *= 1.5;
	}

	float3 tex = float3(inputTexture.sample(quadSampler, clamp(uv, 0.0, 1.0)).rgb);
	tex.r = androidVhsGetYPosCubicBezier(tex.r);
	tex.g = androidVhsGetYPosCubicBezier(tex.g);
	tex.b = androidVhsGetYPosCubicBezier(tex.b);

	distortion *= androidVhsEaseInOutBounce(1.0 - fpos.x);
	float3 color = tex + float3(distortion) + grainValue;

	outputTexture.write(half4(half3(clamp(color, 0.0, 1.0)), 1.0), gid);
}

// MARK: - DV Cam / VHS2 HUD Helpers

float2 androidHudClamp01(float2 v) {
	return clamp(v, 0.0, 1.0);
}

float3 androidHudClamp01(float3 v) {
	return clamp(v, 0.0, 1.0);
}

float androidHudRand(float x) {
	return fract(sin(x) * 43758.5453123);
}

float androidHudRand(float2 st) {
	return fract(sin(dot(st, float2(12.9898, 78.233))) * 43758.5453123);
}

float2 androidHudPixelStep(float2 resolution) {
	return float2(1.0 / max(resolution.x, 1.0), 1.0 / max(resolution.y, 1.0));
}

float2 androidHudMapToSafe(float2 localUV, float2 safeOrigin, float2 safeSize) {
	return safeOrigin + localUV * safeSize;
}

constant int kAndroidHudTextDVCAM[8] = { 89, 79, 80, 69, 32, 67, 65, 77 }; // YOPE CAM
constant int kAndroidHudTextPLAY[4] = { 80, 76, 65, 89 }; // PLAY
constant int kAndroidHudTextSEC60[5] = { 54, 48, 83, 69, 67 }; // 60SEC
constant int kAndroidHudTextDVIN[5] = { 68, 86, 32, 73, 78 }; // DV IN
constant int kAndroidHudTextVHS2CAM[7] = { 67, 65, 77, 69, 82, 65, 49 }; // CAMERA1
constant int kAndroidHudTextSOURCE[6] = { 83, 79, 85, 82, 67, 69 }; // SOURCE

int androidHudGlyphRow(int ch, int row) {
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

float androidHudRect(float2 uv, float2 origin, float2 size) {
	float2 p = (uv - origin) / size;
	return step(0.0, p.x) * step(p.x, 1.0) * step(0.0, p.y) * step(p.y, 1.0);
}

float androidHudDrawChar(float2 suvTopLeft, float2 origin, float2 charSize, int ch) {
	float2 p = (suvTopLeft - origin) / charSize;
	if (p.x < 0.0 || p.x >= 1.0 || p.y < 0.0 || p.y >= 1.0) { return 0.0; }
	int gx = int(floor(p.x * 5.0));
	int gy = int(floor(p.y * 7.0));
	gx = clamp(gx, 0, 4);
	gy = clamp(gy, 0, 6);
	int rowBits = androidHudGlyphRow(ch, gy);
	int bit = (rowBits >> (4 - gx)) & 1;
	return float(bit);
}

float androidHudDrawText(
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
		mask = max(mask, androidHudDrawChar(suvTopLeft, cOrigin, charSize, chars[i]));
	}
	return mask;
}

float androidHudDrawTimerHHMMSSFF(
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
		mask = max(mask, androidHudDrawChar(suvTopLeft, cOrigin, charSize, chars[i]));
	}
	return mask;
}

float androidHudDrawTimerMMSSFF(
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
		mask = max(mask, androidHudDrawChar(suvTopLeft, cOrigin, charSize, chars[i]));
	}
	return mask;
}

// MARK: - DV Cam (with HUD)

float androidHudDvcamStripes(float t, float e) {
	return smoothstep((1.0 - e) / 4.0, (1.0 + e) / 4.0, abs(fract(t) - 0.5));
}

float2 androidHudDvcamFuzzyGrid(float2 c, float e) {
	return floor(c) + smoothstep(1.0 - e, 1.0, fract(c));
}

float androidHudDvcamGlitch1(float t) {
	return step(1.5 - 1.0 / 20.0, fmod(t, 1.5)) + step(2.7 - 1.0 / 20.0, fmod(t, 2.7)) + step(3.9 - 1.0 / 20.0, fmod(t, 3.9));
}

float androidHudDvcamGlitch2(float t) {
	return step(1.9 - 1.0 / 20.0, fmod(t, 1.9)) + step(2.9 - 1.0 / 20.0, fmod(t, 2.9)) + step(5.9 - 1.0 / 20.0, fmod(t, 5.9));
}

float2 androidHudDvcamHash22(float2 p) {
	p = float2(dot(p, float2(127.1, 311.7)), dot(p, float2(269.5, 183.3)));
	return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
}

kernel void androidDvCamHudKernel(
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

	if (androidHudDvcamGlitch1(*timeSec) > 0.5) {
		float2 gridCoord = androidHudDvcamFuzzyGrid(squareXY * 16.0, 0.15) / 16.0;
		gridCoord = gridCoord * (15.0 / 16.0) + 0.5 / 16.0;
		uv += (androidHudDvcamHash22(gridCoord) * 2.0 - 1.0) * 0.03;
	}

	float3 c = float3(inputTexture.sample(quadSampler, androidHudClamp01(uv)).rgb);
	float blueShift = float(inputTexture.sample(quadSampler, androidHudClamp01(uv + float2(-0.014, 0.002))).b);
	c.b = mix(c.b, blueShift, 0.35);

	if (androidHudDvcamGlitch2(*timeSec) > 0.5) {
		float yvFactor = androidHudDvcamStripes(uv.y * 5.0, 0.25);
		float2 rg = pow(c.rg, mix(float2(1.0, 1.2), float2(0.9, 1.05), yvFactor));
		c.r = rg.x;
		c.g = rg.y;
	}

	float vline = androidHudDvcamStripes(squareXY.x * 160.0 + 3.0 * sin(squareXY.y * 12.0), 0.4);
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
	float2 suvFlip = suv;
	float2 charSizeLocal = float2(0.032, 0.054);
	float spacingLocal = 0.0060;
	float2 charSize = charSizeLocal;
	float spacing = spacingLocal;
	float textWidthTimer = 11.0 * charSize.x + 10.0 * spacing;
	float2 sh = float2(1.0 / imageSize.x, 1.0 / imageSize.y);

	float textWidthTimerLocal = 11.0 * charSizeLocal.x + 10.0 * spacingLocal;
	float2 timerOriginLocal = float2(1.0 - textWidthTimerLocal - 0.03, 0.06);
	float2 playOriginLocal = float2(timerOriginLocal.x - (4.0 * charSizeLocal.x + 3.0 * spacingLocal) - 0.014, timerOriginLocal.y);
	float2 secOriginLocal = float2(1.0 - (5.0 * charSizeLocal.x + 4.0 * spacingLocal) - 0.03, timerOriginLocal.y + charSizeLocal.y + 0.010);
	float2 dvCamOriginLocal = float2(0.03, 1.0 - charSizeLocal.y * 3.0 - 0.07);
	float2 dvInOriginLocal = float2(0.03, dvCamOriginLocal.y + charSizeLocal.y + 0.010);

	float2 timerOrigin = timerOriginLocal;
	float2 playOrigin = playOriginLocal;
	float2 secOrigin = secOriginLocal;
	float2 dvCamOrigin = dvCamOriginLocal;
	float2 dvInOrigin = dvInOriginLocal;

	float textMask = 0.0;
	float shadowMask = 0.0;

	textMask = max(textMask, max(
		androidHudDrawTimerHHMMSSFF(suv, timerOrigin, charSize, spacing, hh, mm, ss, ff),
		androidHudDrawTimerHHMMSSFF(suvFlip, timerOrigin, charSize, spacing, hh, mm, ss, ff)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawTimerHHMMSSFF(suv + sh, timerOrigin, charSize, spacing, hh, mm, ss, ff),
		androidHudDrawTimerHHMMSSFF(suvFlip + sh, timerOrigin, charSize, spacing, hh, mm, ss, ff)
	));

	textMask = max(textMask, max(
		androidHudDrawText(suv, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4),
		androidHudDrawText(suvFlip, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4),
		androidHudDrawText(suvFlip + sh, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4)
	));

	textMask = max(textMask, max(
		androidHudDrawText(suv, secOrigin, charSize, spacing, kAndroidHudTextSEC60, 5),
		androidHudDrawText(suvFlip, secOrigin, charSize, spacing, kAndroidHudTextSEC60, 5)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, secOrigin, charSize, spacing, kAndroidHudTextSEC60, 5),
		androidHudDrawText(suvFlip + sh, secOrigin, charSize, spacing, kAndroidHudTextSEC60, 5)
	));

	textMask = max(textMask, max(
		androidHudDrawText(suv, dvCamOrigin, charSize, spacing, kAndroidHudTextDVCAM, 8),
		androidHudDrawText(suvFlip, dvCamOrigin, charSize, spacing, kAndroidHudTextDVCAM, 8)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, dvCamOrigin, charSize, spacing, kAndroidHudTextDVCAM, 8),
		androidHudDrawText(suvFlip + sh, dvCamOrigin, charSize, spacing, kAndroidHudTextDVCAM, 8)
	));

	textMask = max(textMask, max(
		androidHudDrawText(suv, dvInOrigin, charSize, spacing, kAndroidHudTextDVIN, 5),
		androidHudDrawText(suvFlip, dvInOrigin, charSize, spacing, kAndroidHudTextDVIN, 5)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, dvInOrigin, charSize, spacing, kAndroidHudTextDVIN, 5),
		androidHudDrawText(suvFlip + sh, dvInOrigin, charSize, spacing, kAndroidHudTextDVIN, 5)
	));

	float bgMask = 0.0;
	bgMask = max(bgMask, max(
		androidHudRect(suv, timerOrigin - float2(0.012, 0.008), float2(textWidthTimer + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, timerOrigin - float2(0.012, 0.008), float2(textWidthTimer + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, secOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, secOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, dvCamOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, dvCamOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, dvInOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, dvInOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016))
	));

	c = mix(c, float3(0.02, 0.025, 0.03), clamp(bgMask, 0.0, 1.0) * 0.82);
	c = mix(c, float3(0.0), clamp(shadowMask, 0.0, 1.0) * 0.72);
	c = mix(c, float3(0.96, 0.99, 1.0), clamp(textMask, 0.0, 1.0));

	outputTexture.write(half4(half3(androidHudClamp01(c)), 1.0), gid);
}

kernel void androidVhsDVCamTextHudKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	constant float *hudOpacityIn [[buffer(1)]],
	constant float *counterMirrorIn [[buffer(2)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 imageSize = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / imageSize;
	float3 c = float3(inputTexture.sample(quadSampler, androidHudClamp01(uv)).rgb);

	float aspect = imageSize.x / max(imageSize.y, 1.0);

	int timeMs = int(max(*timeSec, 0.0) * 1000.0);
	int hh = (timeMs / (60 * 60 * 1000)) % 100;
	int mm = (timeMs % (60 * 60 * 1000)) / (60 * 1000);
	int ss = ((timeMs % (60 * 60 * 1000)) % (60 * 1000)) / 1000;
	int ff = (((timeMs % (60 * 60 * 1000)) % (60 * 1000)) % 1000) * 30 / 1000;

	float2 suv = uv;
	if (*counterMirrorIn > 0.5) {
		suv.x = 1.0 - suv.x;
	}
	float2 charSizeLocal = float2(0.030, 0.051);
	float spacingLocal = 0.0055;
	float2 charSize = charSizeLocal;
	float spacing = spacingLocal;
	float2 sh = float2(1.0 / imageSize.x, 1.0 / imageSize.y);
	float sideMargin = 0.050;
	float topMargin = 0.070;
	float bottomMargin = 0.065;

	float textWidthTimerLocal = 11.0 * charSizeLocal.x + 10.0 * spacingLocal;
	float2 timerOriginLocal = float2(1.0 - textWidthTimerLocal - sideMargin, topMargin);
	float2 playOriginLocal = float2(timerOriginLocal.x - (4.0 * charSizeLocal.x + 3.0 * spacingLocal) - 0.014, timerOriginLocal.y);
	float2 secOriginLocal = float2(1.0 - (5.0 * charSizeLocal.x + 4.0 * spacingLocal) - sideMargin, timerOriginLocal.y + charSizeLocal.y + 0.010);
	float2 dvCamOriginLocal = float2(sideMargin, 1.0 - charSizeLocal.y * 3.0 - bottomMargin);
	float2 dvInOriginLocal = float2(sideMargin, dvCamOriginLocal.y + charSizeLocal.y + 0.010);

	float2 timerOrigin = timerOriginLocal;
	float2 playOrigin = playOriginLocal;
	float2 secOrigin = secOriginLocal;
	float2 dvCamOrigin = dvCamOriginLocal;
	float2 dvInOrigin = dvInOriginLocal;

	float textMask = 0.0;
	float shadowMask = 0.0;

	textMask = max(textMask, androidHudDrawTimerHHMMSSFF(suv, timerOrigin, charSize, spacing, hh, mm, ss, ff));
	shadowMask = max(shadowMask, androidHudDrawTimerHHMMSSFF(suv + sh, timerOrigin, charSize, spacing, hh, mm, ss, ff));

	textMask = max(textMask, androidHudDrawText(suv, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4));
	shadowMask = max(shadowMask, androidHudDrawText(suv + sh, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4));

	textMask = max(textMask, androidHudDrawText(suv, secOrigin, charSize, spacing, kAndroidHudTextSEC60, 5));
	shadowMask = max(shadowMask, androidHudDrawText(suv + sh, secOrigin, charSize, spacing, kAndroidHudTextSEC60, 5));

	textMask = max(textMask, androidHudDrawText(suv, dvCamOrigin, charSize, spacing, kAndroidHudTextDVCAM, 8));
	shadowMask = max(shadowMask, androidHudDrawText(suv + sh, dvCamOrigin, charSize, spacing, kAndroidHudTextDVCAM, 8));

	textMask = max(textMask, androidHudDrawText(suv, dvInOrigin, charSize, spacing, kAndroidHudTextDVIN, 5));
	shadowMask = max(shadowMask, androidHudDrawText(suv + sh, dvInOrigin, charSize, spacing, kAndroidHudTextDVIN, 5));

	float textWidthTimer = 11.0 * charSize.x + 10.0 * spacing;
	float bgMask = 0.0;
	bgMask = max(bgMask, androidHudRect(suv, timerOrigin - float2(0.012, 0.008), float2(textWidthTimer + 0.024, charSize.y + 0.016)));
	bgMask = max(bgMask, androidHudRect(suv, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016)));
	bgMask = max(bgMask, androidHudRect(suv, secOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016)));
	bgMask = max(bgMask, androidHudRect(suv, dvCamOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016)));
	bgMask = max(bgMask, androidHudRect(suv, dvInOrigin - float2(0.012, 0.008), float2(5.0 * charSize.x + 4.0 * spacing + 0.024, charSize.y + 0.016)));

	float2 recOrigin = float2(sideMargin, topMargin);
	float2 recP = (suv - recOrigin) * float2(aspect, 1.0);
	float recDot = smoothstep(0.012, 0.010, length(recP));
	float recBlink = step(0.5, fract(*timeSec * 2.0));

	float hudOpacity = clamp(*hudOpacityIn, 0.0, 1.0);
	c = mix(c, float3(0.02, 0.025, 0.03), clamp(bgMask, 0.0, 1.0) * 0.78 * hudOpacity);
	c = mix(c, float3(0.0), clamp(shadowMask, 0.0, 1.0) * 0.68 * hudOpacity);
	c = mix(c, float3(0.96, 0.99, 1.0), clamp(textMask, 0.0, 1.0) * hudOpacity);
	c = mix(c, float3(1.0, 0.20, 0.10), recDot * recBlink * 0.9 * hudOpacity);

	outputTexture.write(half4(half3(androidHudClamp01(c)), 1.0), gid);
}

// MARK: - VHS2 (with HUD)

float androidHudVhs2Grain(float2 uv, float timeSec) {
	float x = (uv.x + 4.0) * (uv.y + 4.0) * (timeSec * 10.0);
	float g = (fmod((fmod(x, 13.0) + 1.0) * (fmod(x, 123.0) + 1.0), 0.01) - 0.005) * 14.0;
	return g;
}

float androidHudVhs2EaseInOutBounce(float t) {
	if (t < 0.5) {
		return 8.0 * pow(2.0, 8.0 * (t - 1.0)) * abs(sin(t * 3.14 * 7.0));
	}
	return 1.0 - 8.0 * pow(2.0, -8.0 * t) * abs(sin(t * 3.14 * 7.0));
}

float androidHudVhs2BezierY(float x) {
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

kernel void androidVhs2HudKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant float *timeSec [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) { return; }
	constexpr sampler quadSampler(filter::linear, address::clamp_to_edge);
	float2 resolution = float2(outputTexture.get_width(), outputTexture.get_height());
	float2 uv = (float2(gid) + 0.5) / resolution;
	float aspect = resolution.x / max(resolution.y, 1.0);
	float2 safeOrigin = float2(0.0, 0.0);
	float2 safeSize = float2(1.0, 1.0);
	if (aspect < 1.0) {
		safeOrigin.y = 0.5 * (1.0 - aspect);
		safeSize.y = aspect;
	} else {
		float invAspect = 1.0 / aspect;
		safeOrigin.x = 0.5 * (1.0 - invAspect);
		safeSize.x = invAspect;
	}

	float2 st = uv;
	float time = *timeSec;
	float2 grid = float2(androidHudRand(time * floor(uv.y * 250.0)) * androidHudRand(time) * 36.0, 1.0);
	if (grid.x < 0.0) { grid.x = 1.0; }
	st *= grid;
	float2 ipos = floor(st);
	float2 fpos = fract(st);

	float distortion = 0.0;
	float value = androidHudRand(ipos.yx + grid * 2000.0);
	value *= value;
	float grain = androidHudVhs2Grain(uv, time);
	if (1.0 - uv.y > 0.1 && 1.0 - uv.y < 0.97) {
		if (value > 0.95) { distortion = 0.6; }
		value = androidHudRand(grid);
		if (value > 0.05) { distortion = 0.0; }
	} else {
		if (value > 0.92) { distortion = 0.6; }
		grain *= 1.5;
	}

	float2 pix = androidHudPixelStep(resolution);
	float3 tex =
		5.0 * float3(inputTexture.sample(quadSampler, uv).rgb) -
		float3(inputTexture.sample(quadSampler, androidHudClamp01(uv + float2(pix.x, 0.0))).rgb) -
		float3(inputTexture.sample(quadSampler, androidHudClamp01(uv - float2(pix.x, 0.0))).rgb) -
		float3(inputTexture.sample(quadSampler, androidHudClamp01(uv + float2(0.0, pix.y))).rgb) -
		float3(inputTexture.sample(quadSampler, androidHudClamp01(uv - float2(0.0, pix.y))).rgb);

	tex.r = androidHudVhs2BezierY(tex.r);
	tex.g = androidHudVhs2BezierY(tex.g);
	tex.b = androidHudVhs2BezierY(tex.b);

	distortion *= androidHudVhs2EaseInOutBounce(1.0 - fpos.x);
	float3 color = tex + float3(distortion) + grain * 0.4;

	int timeMs = int(max(*timeSec, 0.0) * 1000.0);
	int mm = (timeMs % (60 * 60 * 1000)) / (60 * 1000);
	int ss = ((timeMs % (60 * 60 * 1000)) % (60 * 1000)) / 1000;
	int ff = (((timeMs % (60 * 60 * 1000)) % (60 * 1000)) % 1000) * 30 / 1000;

	float2 screenUV = (float2(gid) + 0.5) / resolution;
	float2 suv = screenUV;
	float2 suvFlip = suv;
	float2 charSizeLocal = float2(0.034, 0.056);
	float spacingLocal = 0.0062;
	float2 charSize = charSizeLocal;
	float spacing = spacingLocal;
	float2 sh = float2(1.0 / resolution.x, 1.0 / resolution.y);

	float2 camOriginLocal = float2(0.03, 0.07);
	float2 playOriginLocal = float2(0.03, camOriginLocal.y + charSizeLocal.y + 0.010);
	float2 timerOriginLocal = float2(0.03, playOriginLocal.y + charSizeLocal.y + 0.010);
	float2 sourceOriginLocal = float2(1.0 - (6.0 * charSizeLocal.x + 5.0 * spacingLocal) - 0.03, camOriginLocal.y);

	float2 camOrigin = androidHudMapToSafe(camOriginLocal, safeOrigin, safeSize);
	float2 playOrigin = androidHudMapToSafe(playOriginLocal, safeOrigin, safeSize);
	float2 timerOrigin = androidHudMapToSafe(timerOriginLocal, safeOrigin, safeSize);
	float2 sourceOrigin = androidHudMapToSafe(sourceOriginLocal, safeOrigin, safeSize);

	float textMask = 0.0;
	float shadowMask = 0.0;

	textMask = max(textMask, max(
		androidHudDrawText(suv, camOrigin, charSize, spacing, kAndroidHudTextVHS2CAM, 7),
		androidHudDrawText(suvFlip, camOrigin, charSize, spacing, kAndroidHudTextVHS2CAM, 7)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, camOrigin, charSize, spacing, kAndroidHudTextVHS2CAM, 7),
		androidHudDrawText(suvFlip + sh, camOrigin, charSize, spacing, kAndroidHudTextVHS2CAM, 7)
	));

	textMask = max(textMask, max(
		androidHudDrawText(suv, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4),
		androidHudDrawText(suvFlip, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4),
		androidHudDrawText(suvFlip + sh, playOrigin, charSize, spacing, kAndroidHudTextPLAY, 4)
	));

	textMask = max(textMask, max(
		androidHudDrawTimerMMSSFF(suv, timerOrigin, charSize, spacing, mm, ss, ff),
		androidHudDrawTimerMMSSFF(suvFlip, timerOrigin, charSize, spacing, mm, ss, ff)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawTimerMMSSFF(suv + sh, timerOrigin, charSize, spacing, mm, ss, ff),
		androidHudDrawTimerMMSSFF(suvFlip + sh, timerOrigin, charSize, spacing, mm, ss, ff)
	));

	textMask = max(textMask, max(
		androidHudDrawText(suv, sourceOrigin, charSize, spacing, kAndroidHudTextSOURCE, 6),
		androidHudDrawText(suvFlip, sourceOrigin, charSize, spacing, kAndroidHudTextSOURCE, 6)
	));
	shadowMask = max(shadowMask, max(
		androidHudDrawText(suv + sh, sourceOrigin, charSize, spacing, kAndroidHudTextSOURCE, 6),
		androidHudDrawText(suvFlip + sh, sourceOrigin, charSize, spacing, kAndroidHudTextSOURCE, 6)
	));

	float bgMask = 0.0;
	bgMask = max(bgMask, max(
		androidHudRect(suv, camOrigin - float2(0.012, 0.008), float2(7.0 * charSize.x + 6.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, camOrigin - float2(0.012, 0.008), float2(7.0 * charSize.x + 6.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, playOrigin - float2(0.012, 0.008), float2(4.0 * charSize.x + 3.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, timerOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, timerOrigin - float2(0.012, 0.008), float2(8.0 * charSize.x + 7.0 * spacing + 0.024, charSize.y + 0.016))
	));
	bgMask = max(bgMask, max(
		androidHudRect(suv, sourceOrigin - float2(0.012, 0.008), float2(6.0 * charSize.x + 5.0 * spacing + 0.024, charSize.y + 0.016)),
		androidHudRect(suvFlip, sourceOrigin - float2(0.012, 0.008), float2(6.0 * charSize.x + 5.0 * spacing + 0.024, charSize.y + 0.016))
	));

	color = mix(color, float3(0.02, 0.025, 0.03), clamp(bgMask, 0.0, 1.0) * 0.82);
	color = mix(color, float3(0.0), clamp(shadowMask, 0.0, 1.0) * 0.72);
	color = mix(color, float3(0.96, 0.99, 1.0), clamp(textMask, 0.0, 1.0));

	outputTexture.write(half4(half3(androidHudClamp01(color)), 1.0), gid);
}
