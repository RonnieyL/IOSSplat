

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
    kGaussianUniforms = 3,
};

struct RGBUniforms {
    matrix_float3x3 viewToCamera;
    float viewRatio;
    float radius;
};

struct PointCloudUniforms {
    matrix_float4x4 viewProjectionMatrix;
    matrix_float4x4 localToWorld;
    matrix_float3x3 cameraIntrinsicsInversed;
    simd_float2 cameraResolution;
    
    float particleSize;
    int maxPoints;
    int pointCloudCurrentIndex;
    int confidenceThreshold;
};

struct ParticleUniforms {
    simd_float3 position;
    simd_float3 color;
    float confidence;
};

// GaussianSplat struct - only for Metal shaders
// Swift will use SplatRenderer.Splat type directly (defined in Swift)
#ifdef __METAL_VERSION__
struct GaussianUniforms {
    packed_float3 position;    // 3D world position (12 bytes, offset 0)
    packed_half4 color;        // RGBA in Float16 (8 bytes, offset 12)
    packed_half3 covA;         // 3D covariance upper triangle (6 bytes, offset 20)
    packed_half3 covB;         // 3D covariance lower triangle (6 bytes, offset 26)
    // Total: 32 bytes (reduced from 56 bytes - more efficient!)
};
#endif

#endif /* ShaderTypes_h */
