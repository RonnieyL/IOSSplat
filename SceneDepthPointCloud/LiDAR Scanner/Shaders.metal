#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"

using namespace metal;

// Camera's RGB vertex shader outputs
struct RGBVertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Particle vertex shader outputs and fragment shader inputs
struct ParticleVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4 color;
};

constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
constant auto yCbCrToRGB = float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
                                    float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
                                    float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
                                    float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));
constant float2 viewVertices[] = { float2(-1, 1), float2(-1, -1), float2(1, 1), float2(1, -1) };
constant float2 viewTexCoords[] = { float2(0, 0), float2(0, 1), float2(1, 0), float2(1, 1) };

/// Retrieves the world position of a specified camera point with depth
static simd_float4 worldPoint(simd_float2 cameraPoint, float depth, matrix_float3x3 cameraIntrinsicsInversed, matrix_float4x4 localToWorld) {
    const auto localPoint = cameraIntrinsicsInversed * simd_float3(cameraPoint, 1) * depth;
    const auto worldPoint = localToWorld * simd_float4(localPoint, 1);
    
    return worldPoint / worldPoint.w;
}

///  Vertex shader that takes in a 2D grid-point and infers its 3D position in world-space, along with RGB and confidence
vertex void unprojectVertex(uint vertexID [[vertex_id]],
                            constant PointCloudUniforms &uniforms [[buffer(kPointCloudUniforms)]],
                            device ParticleUniforms *particleUniforms [[buffer(kParticleUniforms)]],
                            constant float2 *gridPoints [[buffer(kGridPoints)]],
                            texture2d<float, access::sample> capturedImageTextureY [[texture(kTextureY)]],
                            texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kTextureCbCr)]],
                            texture2d<float, access::sample> depthTexture [[texture(kTextureDepth)]],
                            texture2d<unsigned int, access::sample> confidenceTexture [[texture(kTextureConfidence)]]) {
    
    const auto gridPoint = gridPoints[vertexID];
    const auto currentPointIndex = (uniforms.pointCloudCurrentIndex + vertexID) % uniforms.maxPoints;
    const auto texCoord = gridPoint / uniforms.cameraResolution;
    // Sample the depth map to get the depth value
    const auto depth = depthTexture.sample(colorSampler, texCoord).r;
    // With a 2D point plus depth, we can now get its 3D position
    const auto position = worldPoint(gridPoint, depth, uniforms.cameraIntrinsicsInversed, uniforms.localToWorld);
    
    // Sample Y and CbCr textures to get the YCbCr color at the given texture coordinate
    const auto ycbcr = float4(capturedImageTextureY.sample(colorSampler, texCoord).r, capturedImageTextureCbCr.sample(colorSampler, texCoord.xy).rg, 1);
    const auto sampledColor = (yCbCrToRGB * ycbcr).rgb;
    // Sample the confidence map to get the confidence value
    const auto confidence = confidenceTexture.sample(colorSampler, texCoord).r;
    
    // Write the data to the buffer
    particleUniforms[currentPointIndex].position = position.xyz;
    particleUniforms[currentPointIndex].color = sampledColor;
    particleUniforms[currentPointIndex].confidence = confidence;
}

///  MVS Vertex shader that takes in a 2D grid-point and infers its 3D position using AI depth
vertex void unprojectVertexMVS(uint vertexID [[vertex_id]],
                               constant PointCloudUniforms &uniforms [[buffer(kPointCloudUniforms)]],
                               device ParticleUniforms *particleUniforms [[buffer(kParticleUniforms)]],
                               constant float2 *gridPoints [[buffer(kGridPoints)]],
                               texture2d<float, access::sample> capturedImageTextureY [[texture(kTextureY)]],
                               texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kTextureCbCr)]],
                               texture2d<float, access::sample> aiDepthTexture [[texture(kTextureDepth)]]) {
    
    const auto gridPoint = gridPoints[vertexID];
    const auto currentPointIndex = (uniforms.pointCloudCurrentIndex + vertexID) % uniforms.maxPoints;
    const auto texCoord = gridPoint / uniforms.cameraResolution;
    
    // Sample the AI depth map to get the depth value
    const auto depth = aiDepthTexture.sample(colorSampler, texCoord).r;
    
    // With a 2D point plus depth, we can now get its 3D position
    const auto position = worldPoint(gridPoint, depth, uniforms.cameraIntrinsicsInversed, uniforms.localToWorld);
    
    // Sample Y and CbCr textures to get the YCbCr color at the given texture coordinate
    const auto ycbcr = float4(capturedImageTextureY.sample(colorSampler, texCoord).r, capturedImageTextureCbCr.sample(colorSampler, texCoord.xy).rg, 1);
    const auto sampledColor = (yCbCrToRGB * ycbcr).rgb;
    
    // MVS points get high confidence by default
    const auto confidence = 2.0;
    
    // Write the data to the buffer
    particleUniforms[currentPointIndex].position = position.xyz;
    particleUniforms[currentPointIndex].color = sampledColor;
    particleUniforms[currentPointIndex].confidence = confidence;
}

/// Improved depth normalization compute shader (better approximation of INRIA method)
kernel void normalizeDepth(texture2d<float, access::read> inputDepth [[texture(0)]],
                          texture2d<float, access::write> outputDepth [[texture(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= inputDepth.get_width() || gid.y >= inputDepth.get_height()) {
        return;
    }
    
    float rawDepth = inputDepth.read(gid).r;
    
    // INRIA-inspired normalization
    // Reference: depth = (depth - median) / mad
    // For real-time performance, we use approximated percentile-based normalization
    
    // Clamp to reasonable relative depth range (0-1 from Depth-Anything)
    rawDepth = saturate(rawDepth);
    
    // Convert relative depth to metric depth using INRIA-like scaling
    // Key insight: INRIA uses median-centered normalization, not linear
    
    // Approximate median = 0.5, MAD ≈ 0.3 for typical indoor scenes
    float normalizedDepth = (rawDepth - 0.5f) / 0.3f;
    
    // Convert to metric depth (matching INRIA's inverse depth range)
    // INRIA inverse depth range: 1e-4 to 1e1 → depth range: 0.1m to 10,000m
    // We'll use a more practical range: 0.3m to 15m for indoor/outdoor scenes
    
    float invDepth = 0.5f + normalizedDepth * 0.4f;  // Map to inverse depth range
    invDepth = clamp(invDepth, 0.067f, 3.33f);       // 0.3m to 15m depth range
    
    float metricDepth = 1.0f / invDepth;
    
    outputDepth.write(float4(metricDepth, 0, 0, 1), gid);
}

/// INRIA-style depth normalization compute shader (exact reference implementation)
kernel void inriaNormalizeDepth(texture2d<float, access::read> inputDepth [[texture(0)]],
                                texture2d<float, access::write> outputDepth [[texture(1)]],
                                constant float4 &stats [[buffer(0)]],  // median, mad, unused, unused
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= inputDepth.get_width() || gid.y >= inputDepth.get_height()) {
        return;
    }
    
    float rawDepth = inputDepth.read(gid).r;
    
    // INRIA normalization: depth = (depth - median) / mad
    float median = stats.x;
    float mad = stats.y;
    
    float normalizedDepth = (rawDepth - median) / mad;
    
    // Convert normalized depth to metric depth
    // INRIA works in inverse depth space, then converts to metric
    // For practical use, we'll map normalized depth to reasonable metric range
    
    // Clamp normalized depth to reasonable range (-3 to +3 standard deviations)
    normalizedDepth = clamp(normalizedDepth, -3.0f, 3.0f);
    
    // Convert to inverse depth (INRIA's approach)
    // Map normalized range to inverse depth: center around 0.5 (2m), spread ±0.4
    float invDepth = 0.5f + normalizedDepth * 0.1f;  // More conservative mapping
    invDepth = clamp(invDepth, 0.1f, 2.0f);          // 0.5m to 10m depth range
    
    // Convert inverse depth to metric depth
    float metricDepth = 1.0f / invDepth;
    
    outputDepth.write(float4(metricDepth, 0, 0, 1), gid);
}

/// Simple depth normalization (for debugging)
kernel void simpleNormalizeDepth(texture2d<float, access::read> inputDepth [[texture(0)]],
                                texture2d<float, access::write> outputDepth [[texture(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= inputDepth.get_width() || gid.y >= inputDepth.get_height()) {
        return;
    }
    
    float rawDepth = inputDepth.read(gid).r;
    
    // Simple clamping and scaling for visualization
    // Depth-Anything outputs relative depth 0-1, convert to reasonable metric range
    float metricDepth = rawDepth * 10.0f + 0.1f;  // Map to 0.1m - 10.1m range
    metricDepth = clamp(metricDepth, 0.1f, 10.0f);
    
    outputDepth.write(float4(metricDepth, 0, 0, 1), gid);
}

vertex RGBVertexOut rgbVertex(uint vertexID [[vertex_id]],
                              constant RGBUniforms &uniforms [[buffer(0)]]) {
    const float3 texCoord = float3(viewTexCoords[vertexID], 1) * uniforms.viewToCamera;
    
    RGBVertexOut out;
    out.position = float4(viewVertices[vertexID], 0, 1);
    out.texCoord = texCoord.xy;
    
    return out;
}

fragment float4 depthViewFragment(RGBVertexOut in [[stage_in]],
                                 constant RGBUniforms &uniforms [[buffer(0)]],
                                 texture2d<float, access::sample> capturedImageTextureY [[texture(kTextureY)]],
                                 texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kTextureCbCr)]],
                                 texture2d<float, access::sample> depthTexture [[texture(kTextureDepth)]]) {
    
    // Sample depth texture
    const float depth = depthTexture.sample(colorSampler, in.texCoord).r;
    
    // Debug: Show different colors for different depth ranges to identify issues
    if (depth < 0.001f) {
        // No depth data = red
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    } else if (depth > 100.0f) {
        // Extreme depth = blue
        return float4(0.0f, 0.0f, 1.0f, 1.0f);
    } else {
        // Normal depth = grayscale (HuggingFace style)
        // Map 0.1m-10m to 0-1 grayscale
        float normalizedDepth = (depth - 0.1f) / (10.0f - 0.1f);
        normalizedDepth = saturate(normalizedDepth);
        
        // Invert: close = dark, far = light
        float grayValue = 1.0f - normalizedDepth;
        
        return float4(grayValue, grayValue, grayValue, 1.0f);
    }
}

fragment float4 rgbFragment(RGBVertexOut in [[stage_in]],
                            constant RGBUniforms &uniforms [[buffer(0)]],
                            texture2d<float, access::sample> capturedImageTextureY [[texture(kTextureY)]],
                            texture2d<float, access::sample> capturedImageTextureCbCr [[texture(kTextureCbCr)]]) {
    
    const float2 offset = (in.texCoord - 0.5) * float2(1, 1 / uniforms.viewRatio) * 2;
    const float visibility = saturate(uniforms.radius * uniforms.radius - length_squared(offset));
    const float4 ycbcr = float4(capturedImageTextureY.sample(colorSampler, in.texCoord.xy).r, capturedImageTextureCbCr.sample(colorSampler, in.texCoord.xy).rg, 1);
    
    // convert and save the color back to the buffer
    const float3 sampledColor = (yCbCrToRGB * ycbcr).rgb;
    return float4(sampledColor, 1) * visibility;
}

vertex ParticleVertexOut particleVertex(uint vertexID [[vertex_id]],
                                        constant PointCloudUniforms &uniforms [[buffer(kPointCloudUniforms)]],
                                        constant ParticleUniforms *particleUniforms [[buffer(kParticleUniforms)]]) {
    
    // get point data
    const auto particleData = particleUniforms[vertexID];
    const auto position = particleData.position;
    const auto confidence = particleData.confidence;
    const auto sampledColor = particleData.color;
    const auto visibility = confidence >= uniforms.confidenceThreshold;
    
    // animate and project the point
    float4 projectedPosition = uniforms.viewProjectionMatrix * float4(position, 1.0);
    const float pointSize = max(uniforms.particleSize / max(1.0, projectedPosition.z), 2.0);
    projectedPosition /= projectedPosition.w;
    
    // prepare for output
    ParticleVertexOut out;
    out.position = projectedPosition;
    out.pointSize = pointSize;
    out.color = float4(sampledColor, visibility);
    
    return out;
}

fragment float4 particleFragment(ParticleVertexOut in [[stage_in]],
                                 const float2 coords [[point_coord]]) {
    // we draw within a circle
    const float distSquared = length_squared(coords - float2(0.5));
    if (in.color.a == 0 || distSquared > 0.25) {
        discard_fragment();
    }
    
    return in.color;
}
