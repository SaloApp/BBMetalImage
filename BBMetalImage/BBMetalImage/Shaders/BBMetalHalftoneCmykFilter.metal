//
//  BBMetalHalftoneCmykFilter.metal
//  BBMetalImage
//

#include <metal_stdlib>
#include "BBMetalShaderTypes.h"
using namespace metal;

struct HalftoneParams {
	float size;
	float softness;
	float grainSize;
	float contrast;
	float grainMixer;
	float grainOverlay;
	float gridNoise;
	float floodC;
	float floodM;
	float floodY;
	float floodK;
	float gainC;
	float gainM;
	float gainY;
	float gainK;
	float type;
	float mixOriginal;
	float4 colorBack;
	float4 colorC;
	float4 colorM;
	float4 colorY;
	float4 colorK;
};

float3 hash23(float2 p) {
	float3 p3 = fract(float3(p.x, p.y, p.x) * float3(0.3183099, 0.3678794, 0.3141592)) + 0.1;
	p3 += dot(p3, p3.yzx + 19.19);
	return fract(float3(p3.x * p3.y, p3.y * p3.z, p3.z * p3.x));
}

float2 randomRG(float2 p) {
	float3 h = hash23(p);
	return h.xy;
}

float sst(float edge0, float edge1, float x) {
	return smoothstep(edge0, edge1, x);
}

float3 valueNoise3(float2 st) {
	float2 i = floor(st);
	float2 f = fract(st);
	float3 a = hash23(i);
	float3 b = hash23(i + float2(1.0, 0.0));
	float3 c = hash23(i + float2(0.0, 1.0));
	float3 d = hash23(i + float2(1.0, 1.0));
	float2 u = f * f * (3.0 - 2.0 * f);
	float3 x1 = mix(a, b, u.x);
	float3 x2 = mix(c, d, u.x);
	return mix(x1, x2, u.y);
}

float getUvFrame(float2 uv, float2 pad) {
	float left = smoothstep(-pad.x, 0.0, uv.x);
	float right = smoothstep(1.0 + pad.x, 1.0, uv.x);
	float bottom = smoothstep(-pad.y, 0.0, uv.y);
	float top = smoothstep(1.0 + pad.y, 1.0, uv.y);
	return left * right * bottom * top;
}

float3 applyContrast(float3 rgb, float contrast) {
	return clamp((rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
}

float getCyan(float4 rgba, float contrast) {
	float3 c = clamp((rgba.rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
	float maxRGB = max(max(c.r, c.g), c.b);
	return (maxRGB > 1e-5 ? (maxRGB - c.r) / maxRGB : 0.0) * rgba.a;
}

float getMagenta(float4 rgba, float contrast) {
	float3 c = clamp((rgba.rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
	float maxRGB = max(max(c.r, c.g), c.b);
	return (maxRGB > 1e-5 ? (maxRGB - c.g) / maxRGB : 0.0) * rgba.a;
}

float getYellow(float4 rgba, float contrast) {
	float3 c = clamp((rgba.rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
	float maxRGB = max(max(c.r, c.g), c.b);
	return (maxRGB > 1e-5 ? (maxRGB - c.b) / maxRGB : 0.0) * rgba.a;
}

float getBlack(float4 rgba, float contrast) {
	float3 c = clamp((rgba.rgb - 0.5) * contrast + 0.5, 0.0, 1.0);
	return (1.0 - max(max(c.r, c.g), c.b)) * rgba.a;
}

float2 cellCenterPos(float2 uv, float2 cellOffset, float channelIdx, float gridNoise) {
	float2 cellCenter = floor(uv) + 0.5 + cellOffset;
	return cellCenter + (randomRG(cellCenter + channelIdx * 50.0) - 0.5) * gridNoise;
}

float2 gridToImageUV(float2 cellCenter, float cosA, float sinA, float shift, float2 pad) {
	float2 uvGrid = float2(
		cosA * (cellCenter.x - shift) - sinA * (cellCenter.y - shift),
		sinA * (cellCenter.x - shift) + cosA * (cellCenter.y - shift)
	);
	return uvGrid * pad + 0.5;
}

float colorMask(
	float2 pos,
	float2 cellCenter,
	float rad,
	float transparency,
	float grain,
	float channelAddon,
	float channelGain,
	float generalComp,
	float softness,
	bool isJoined
) {
	float dist = length(pos - cellCenter);
	float radius = rad;
	radius *= (1.0 + generalComp);
	radius += (0.15 + channelGain * radius);
	radius = max(0.0, radius);
	radius = mix(0.0, radius, transparency);
	radius += channelAddon;
	radius *= (1.0 - grain);

	float mask = 1.0 - sst(0.0, radius, dist);
	if (isJoined) {
		mask = pow(mask, 1.2);
	} else {
		mask = sst(0.5 - 0.5 * softness, 0.51 + 0.49 * softness, mask);
	}

	mask *= mix(1.0, mix(0.5, 1.0, 1.5 * radius), softness);
	return mask;
}

float3 applyInk(float3 paper, float3 inkColor, float cov) {
	float3 inkEffect = mix(float3(1.0), inkColor, clamp(cov, 0.0, 1.0));
	return paper * inkEffect;
}

kernel void halftoneCmykKernel(
	texture2d<half, access::write> outputTexture [[texture(0)]],
	texture2d<half, access::sample> inputTexture [[texture(1)]],
	constant HalftoneParams &params [[buffer(0)]],
	uint2 gid [[thread_position_in_grid]]
) {
	if ((gid.x >= outputTexture.get_width()) || (gid.y >= outputTexture.get_height())) { return; }

	const float2 outSize = float2(outputTexture.get_width(), outputTexture.get_height());
	const float2 uv = (float2(gid) + 0.5) / outSize;

	constexpr sampler quadSampler(mag_filter::linear, min_filter::linear);

	const float imageAspectRatio = float(inputTexture.get_width()) / float(inputTexture.get_height());

	const float shiftC = -0.5;
	const float shiftM = -0.25;
	const float shiftY = 0.2;
	const float shiftK = 0.0;

	const float cosC = 0.9659258;
	const float sinC = 0.2588190;
	const float cosM = 0.2588190;
	const float sinM = 0.9659258;
	const float cosY = 1.0;
	const float sinY = 0.0;
	const float cosK = 0.7071068;
	const float sinK = 0.7071068;

	float cellsPerSide = mix(400.0, 7.0, pow(params.size, 0.7));
	float cellSizeY = 1.0 / cellsPerSide;
	float2 pad = cellSizeY * float2(1.0 / imageAspectRatio, 1.0);
	float2 uvGrid = (uv - 0.5) / pad;
	float insideImageBox = getUvFrame(uv, pad);
	const float borderSize = 0.02;
	float2 edge = min(uv, 1.0 - uv);
	float borderMask = smoothstep(0.0, borderSize, min(edge.x, edge.y));
	insideImageBox *= borderMask;

	float generalComp = 0.1 * params.softness + 0.1 * params.gridNoise + 0.1 * (1.0 - step(0.5, params.type)) * (1.5 - params.softness);

	float2 uvC = float2(cosC * uvGrid.x + sinC * uvGrid.y, -sinC * uvGrid.x + cosC * uvGrid.y) + shiftC;
	float2 uvM = float2(cosM * uvGrid.x + sinM * uvGrid.y, -sinM * uvGrid.x + cosM * uvGrid.y) + shiftM;
	float2 uvY = float2(cosY * uvGrid.x + sinY * uvGrid.y, -sinY * uvGrid.x + cosY * uvGrid.y) + shiftY;
	float2 uvK = float2(cosK * uvGrid.x + sinK * uvGrid.y, -sinK * uvGrid.x + cosK * uvGrid.y) + shiftK;

	float2 grainSize = mix(2000.0, 200.0, params.grainSize) * float2(1.0, 1.0 / imageAspectRatio);
	float2 grainUV = (uv - 0.5) * grainSize + 0.5;
	float3 noiseValues = valueNoise3(grainUV);
	float grain = sst(0.55, 1.0, noiseValues.r);
	grain *= params.grainMixer;

	float cMask = 0.0;
	float mMask = 0.0;
	float yMask = 0.0;
	float kMask = 0.0;
	bool isJoined = params.type > 0.5;

	if (params.type < 1.5) {
		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				float2 cellOffset = float2(float(dx), float(dy));

				float2 cellCenterC = cellCenterPos(uvC, cellOffset, 0.0, params.gridNoise);
				float4 texC = float4(inputTexture.sample(quadSampler, gridToImageUV(cellCenterC, cosC, sinC, shiftC, pad)));
				float cVal = getCyan(texC, params.contrast);
				cMask += colorMask(uvC, cellCenterC, cVal, insideImageBox * texC.a, grain, params.floodC, params.gainC, generalComp, params.softness, isJoined);

				float2 cellCenterM = cellCenterPos(uvM, cellOffset, 1.0, params.gridNoise);
				float4 texM = float4(inputTexture.sample(quadSampler, gridToImageUV(cellCenterM, cosM, sinM, shiftM, pad)));
				float mVal = getMagenta(texM, params.contrast);
				mMask += colorMask(uvM, cellCenterM, mVal, insideImageBox * texM.a, grain, params.floodM, params.gainM, generalComp, params.softness, isJoined);

				float2 cellCenterY = cellCenterPos(uvY, cellOffset, 2.0, params.gridNoise);
				float4 texY = float4(inputTexture.sample(quadSampler, gridToImageUV(cellCenterY, cosY, sinY, shiftY, pad)));
				float yVal = getYellow(texY, params.contrast);
				yMask += colorMask(uvY, cellCenterY, yVal, insideImageBox * texY.a, grain, params.floodY, params.gainY, generalComp, params.softness, isJoined);

				float2 cellCenterK = cellCenterPos(uvK, cellOffset, 3.0, params.gridNoise);
				float4 texK = float4(inputTexture.sample(quadSampler, gridToImageUV(cellCenterK, cosK, sinK, shiftK, pad)));
				float kVal = getBlack(texK, params.contrast);
				kMask += colorMask(uvK, cellCenterK, kVal, insideImageBox * texK.a, grain, params.floodK, params.gainK, generalComp, params.softness, isJoined);
			}
		}
	} else {
		float4 tex = float4(inputTexture.sample(quadSampler, uv));
		tex.rgb = applyContrast(tex.rgb, params.contrast);
		insideImageBox *= tex.a;

		float maxRGB = max(max(tex.r, tex.g), tex.b);
		float k = 1.0 - maxRGB;
		float denom = 1.0 - k;
		float3 cmy = float3(0.0);
		if (denom > 1e-5) {
			cmy = (1.0 - tex.rgb - float3(k)) / denom;
		}
		float4 cmykOriginal = float4(cmy, k) * tex.a;

		for (int dy = -1; dy <= 1; dy++) {
			for (int dx = -1; dx <= 1; dx++) {
				float2 cellOffset = float2(float(dx), float(dy));
				cMask += colorMask(uvC, cellCenterPos(uvC, cellOffset, 0.0, params.gridNoise), cmykOriginal.x, insideImageBox, grain, params.floodC, params.gainC, generalComp, params.softness, isJoined);
				mMask += colorMask(uvM, cellCenterPos(uvM, cellOffset, 1.0, params.gridNoise), cmykOriginal.y, insideImageBox, grain, params.floodM, params.gainM, generalComp, params.softness, isJoined);
				yMask += colorMask(uvY, cellCenterPos(uvY, cellOffset, 2.0, params.gridNoise), cmykOriginal.z, insideImageBox, grain, params.floodY, params.gainY, generalComp, params.softness, isJoined);
				kMask += colorMask(uvK, cellCenterPos(uvK, cellOffset, 3.0, params.gridNoise), cmykOriginal.w, insideImageBox, grain, params.floodK, params.gainK, generalComp, params.softness, isJoined);
			}
		}
	}

	float C = cMask;
	float M = mMask;
	float Y = yMask;
	float K = kMask;

	if (isJoined) {
		float th = 0.5;
		float sLeft = th * params.softness;
		float sRight = (1.0 - th) * params.softness + 0.01;
		float aa = 1.0 / cellsPerSide;
		C = smoothstep(th - sLeft - aa, th + sRight, C);
		M = smoothstep(th - sLeft - aa, th + sRight, M);
		Y = smoothstep(th - sLeft - aa, th + sRight, Y);
		K = smoothstep(th - sLeft - aa, th + sRight, K);
	}

	C *= params.colorC.a;
	M *= params.colorM.a;
	Y *= params.colorY.a;
	K *= params.colorK.a;

	float3 ink = float3(1.0);
	ink = applyInk(ink, params.colorK.rgb, K);
	ink = applyInk(ink, params.colorC.rgb, C);
	ink = applyInk(ink, params.colorM.rgb, M);
	ink = applyInk(ink, params.colorY.rgb, Y);

	float shape = clamp(max(max(C, M), max(Y, K)), 0.0, 1.0);

	float3 color = params.colorBack.rgb * params.colorBack.a;
	float opacity = params.colorBack.a;
	color = mix(color, ink, shape);
	opacity = clamp(opacity + shape, 0.0, 1.0);

	float grainOverlay = mix(noiseValues.g, noiseValues.b, 0.5);
	grainOverlay = pow(grainOverlay, 1.3);
	float grainOverlayV = grainOverlay * 2.0 - 1.0;
	float3 grainOverlayColor = float3(step(0.0, grainOverlayV));
	float grainOverlayStrength = params.grainOverlay * abs(grainOverlayV);
	grainOverlayStrength = pow(grainOverlayStrength, 0.8);
	color = mix(color, grainOverlayColor, 0.5 * grainOverlayStrength);
	opacity = clamp(opacity + 0.5 * grainOverlayStrength, 0.0, 1.0);

	float4 originalSample = float4(inputTexture.sample(quadSampler, uv));
	float3 original = originalSample.rgb;
	color = mix(color, original, clamp(params.mixOriginal, 0.0, 1.0));
	half4 outColor = half4(half(color.r), half(color.g), half(color.b), half(opacity));
	outputTexture.write(outColor, gid);
}
