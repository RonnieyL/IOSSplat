#include <metal_stdlib>
#include <simd/simd.h>
#import "MVSShaderTypes.h"

using namespace metal;

// Camera's RGB vertex shader outputs
struct MVSRGBVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Particle vertex shader outputs and fragment shader inputs
struct MVSParticleVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
};

constexpr sampler mvsColorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
constant auto mvsYCbCrToRGB = float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
                                       float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
                                       float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
                                       float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));
constant float2 mvsViewVertices[] = { float2(-1, 1), float2(-1, -1), float2(1, 1), float2(1, -1) };
constant float2 mvsViewTexCoords[] = { float2(0, 0), float2(0, 1), float2(1, 0), float2(1, 1) };

/// Retrieves the world position of a specified camera point with depth
static simd_float4 mvsWorldPoint(simd_float2 cameraPoint, float depth, matrix_float3x3 cameraIntrinsicsInversed, matrix_float4x4 localToWorld) {
    const auto localPoint = cameraIntrinsicsInversed * simd_float3(cameraPoint, 1) * depth;
    const auto worldPoint = localToWorld * simd_float4(localPoint, 1);
    
    return worldPoint / worldPoint.w;
}

/// Vertex shader for MVS point cloud (simplified - no LiDAR depth/confidence textures)
vertex void mvsUnprojectVertex(uint vertexID [[vertex_id]],
                               constant MVSPointCloudUniforms &uniforms [[buffer(kMVSPointCloudUniforms)]],
                               device MVSParticleUniforms *particleUniforms [[buffer(kMVSParticleUniforms)]],
                               constant float2 *gridPoints [[buffer(kMVSGridPoints)]],
                               texture2d<float, access::sample> capturedImageTextureY [[texture(kMVSTextureY)]],
                               texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kMVSTextureCbCr)]]) {
    
    const auto gridPoint = gridPoints[vertexID];
    const auto currentPointIndex = (uniforms.pointCloudCurrentIndex + vertexID) % uniforms.maxPoints;
    const auto texCoord = gridPoint / uniforms.cameraResolution;
    
    // For MVS mode, we'll generate synthetic depth for now
    // In a real implementation, this would come from MVS depth estimation
    const auto depth = 1.0f + sin(gridPoint.x * 0.01f) * 0.5f; // Synthetic depth
    
    // Sample Y and CbCr textures
    const auto ycbcr = float4(capturedImageTextureY.sample(mvsColorSampler, texCoord).r, capturedImageTextureCbCr.sample(mvsColorSampler, texCoord).rg, 1);
    const auto rgb = (mvsYCbCrToRGB * ycbcr).rgb;
    
    // Transform the 3D point to world coordinates
    const auto worldPos = mvsWorldPoint(gridPoint, depth, uniforms.cameraIntrinsicsInversed, uniforms.cameraToWorld);
    
    // Store the point in the particle buffer
    particleUniforms[currentPointIndex].position = worldPos.xyz;
    particleUniforms[currentPointIndex].color = rgb * 255;
    particleUniforms[currentPointIndex].confidence = 2; // High confidence for MVS points
}

/// RGB background vertex shader
vertex MVSRGBVertexOut mvsRgbVertex(uint vertexID [[vertex_id]],
                                    constant MVSRGBUniforms &uniforms [[buffer(kMVSRGBUniforms)]]) {
    const float2 position = mvsViewVertices[vertexID];
    const float2 texCoord = mvsViewTexCoords[vertexID];
    
    MVSRGBVertexOut out;
    out.position = float4(position, 0, 1);
    out.texCoord = texCoord;
    return out;
}

/// RGB background fragment shader
fragment float4 mvsRgbFragment(MVSRGBVertexOut in [[stage_in]],
                               constant MVSRGBUniforms &uniforms [[buffer(kMVSRGBUniforms)]],
                               texture2d<float, access::sample> capturedImageTextureY [[texture(kMVSTextureY)]],
                               texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kMVSTextureCbCr)]]) {
    
    if (uniforms.radius <= 0.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    
    // Sample Y and CbCr textures
    const auto ycbcr = float4(capturedImageTextureY.sample(mvsColorSampler, in.texCoord).r, 
                             capturedImageTextureCbCr.sample(mvsColorSampler, in.texCoord).rg, 1);
    const auto rgb = (mvsYCbCrToRGB * ycbcr).rgb;
    
    return float4(rgb, uniforms.radius);
}

/// Point cloud vertex shader
vertex MVSParticleVertexOut mvsParticleVertex(uint vertexID [[vertex_id]],
                                              constant MVSPointCloudUniforms &uniforms [[buffer(kMVSPointCloudUniforms)]],
                                              device MVSParticleUniforms *particleUniforms [[buffer(kMVSParticleUniforms)]]) {
    
    // Get the particle data
    const auto particle = particleUniforms[vertexID];
    
    // Transform to clip space
    const auto worldPosition = uniforms.cameraToWorld * float4(particle.position, 1);
    const auto viewPosition = uniforms.worldToCamera * worldPosition;
    
    // Project to screen space (simplified projection)
    const auto clipPosition = float4(viewPosition.xy, viewPosition.z, viewPosition.w);
    
    MVSParticleVertexOut out;
    out.position = clipPosition;
    out.pointSize = uniforms.particleSize;
    out.color = float4(particle.color / 255.0, 1.0);
    
    return out;
}

/// Point cloud fragment shader
fragment float4 mvsParticleFragment(MVSParticleVertexOut in [[stage_in]],
                                    const float2 pointCoord [[point_coord]]) {
    
    // Create circular points
    const float distanceFromCenter = length(pointCoord - float2(0.5));
    if (distanceFromCenter > 0.5) {
        discard_fragment();
    }
    
    return in.color;
}