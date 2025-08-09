
#ifndef ShaderTypes_h
#define ShaderTypes_h

#include <simd/simd.h>

enum TextureIndices {
    kTextureY = 0,
    kTextureCbCr = 1,
    kTextureDepth = 2,
    kTextureConfidence = 3
};

enum BufferIndices {
    kPointCloudUniforms = 0,
    kParticleUniforms = 1,
    kGridPoints = 2,
    kRGBUniforms = 3
};

struct RGBUniforms {
    matrix_float3x3 cameraToWorld;
    matrix_float3x3 viewToCamera;
    float viewRatio;
    float radius;
};

struct PointCloudUniforms {
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

struct ParticleUniforms {
    simd_float3 position;
    simd_float3 color;
    float confidence;
};

#endif /* ShaderTypes_h */
