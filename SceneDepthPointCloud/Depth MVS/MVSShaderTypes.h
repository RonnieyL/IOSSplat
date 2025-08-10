#ifndef MVSShaderTypes_h
#define MVSShaderTypes_h

#include <simd/simd.h>

enum MVSTextureIndices {
    kMVSTextureY = 0,
    kMVSTextureCbCr = 1,
    kMVSTextureDepth = 2,
    kMVSTextureConfidence = 3
};

enum MVSBufferIndices {
    kMVSPointCloudUniforms = 0,
    kMVSParticleUniforms = 1,
    kMVSGridPoints = 2,
    kMVSRGBUniforms = 3
};

struct MVSRGBUniforms {
    matrix_float3x3 cameraToWorld;
    matrix_float3x3 viewToCamera;
    float viewRatio;
    float radius;
};

struct MVSPointCloudUniforms {
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 localToWorld;
    matrix_float4x4 cameraToWorld;
    matrix_float4x4 worldToCamera;
    matrix_float3x3 cameraToWorldRotation;
    matrix_float3x3 cameraIntrinsics;
    matrix_float3x3 cameraIntrinsicsInversed;
    simd_float2 cameraResolution;
    
    float particleSize;
    int maxPoints;
    int pointCloudCurrentIndex;
    int confidenceThreshold;
    int numGridPoints;
};

struct MVSParticleUniforms {
    simd_float3 position;
    simd_float3 color;
    float confidence;
};

#endif /* MVSShaderTypes_h */