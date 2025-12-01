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

vertex RGBVertexOut rgbVertex(uint vertexID [[vertex_id]],
                              constant RGBUniforms &uniforms [[buffer(0)]]) {
    const float3 texCoord = float3(viewTexCoords[vertexID], 1) * uniforms.viewToCamera;
    
    RGBVertexOut out;
    out.position = float4(viewVertices[vertexID], 0, 1);
    out.texCoord = texCoord.xy;
    
    return out;
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

// ============================================================================
// LoG Probability Smoothing Kernel
// ============================================================================

/// Applies Gaussian smoothing to the Laplacian response map
/// This matches the second convolution in get_lapla_norm: F.conv2d(laplacian_norm, kernel)
kernel void smoothLaplacianResponse(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint width = inTexture.get_width();
    const uint height = inTexture.get_height();

    // Bounds check
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    // 5×5 Gaussian kernel (σ ≈ 1.0)
    // Generated using: exp(-(x²+y²)/(2*σ²)) / (2*π*σ²)
    // Normalized so sum = 1
    const float gauss5[25] = {
        0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902,
        0.01330621, 0.05963429, 0.09832033, 0.05963429, 0.01330621,
        0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823,
        0.01330621, 0.05963429, 0.09832033, 0.05963429, 0.01330621,
        0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902
    };

    float sum = 0;

    // Apply 5×5 convolution
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int2 samplePos = int2(gid) + int2(dx, dy);

            // Clamp to texture bounds
            samplePos.x = clamp(samplePos.x, 0, int(width) - 1);
            samplePos.y = clamp(samplePos.y, 0, int(height) - 1);

            float value = inTexture.read(uint2(samplePos)).r;
            int kernelIndex = (dy + 2) * 5 + (dx + 2);
            sum += value * gauss5[kernelIndex];
        }
    }

    outTexture.write(float4(sum, 0, 0, 0), gid);
}

// MARK: - GPU Bernoulli Sampling

/// Simple hash function for pseudo-random number generation
static uint wang_hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

/// Generate a pseudo-random float in [0, 1) from pixel coordinates and frame counter
static float random_float(uint x, uint y, uint frameCounter) {
    uint seed = wang_hash(x + wang_hash(y + wang_hash(frameCounter)));
    return float(seed) / 4294967296.0; // 2^32
}

/// GPU-accelerated Bernoulli sampling kernel
/// Each thread processes one pixel and probabilistically samples it
kernel void bernoulliSample(constant float *probabilityMap [[buffer(kProbabilityMap)]],
                            device float2 *sampledPoints [[buffer(kSampledPoints)]],
                            device atomic_uint *atomicCounter [[buffer(kAtomicCounter)]],
                            constant SamplingUniforms &uniforms [[buffer(kSamplingUniforms)]],
                            uint2 gid [[thread_position_in_grid]]) {
    
    // Boundary check
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }
    
    // Read probability value
    uint index = gid.y * uniforms.stride + gid.x;
    float probability = probabilityMap[index];
    
    // Generate random number (use a simple frame counter that changes each call)
    // Note: We'll pass frame counter via uniforms or use gid as seed
    float randomValue = random_float(gid.x, gid.y, uniforms.stride); // Using stride as pseudo-frame-counter
    
    // Bernoulli test
    if (randomValue < probability) {
        // Atomically claim a slot in the output buffer
        uint outputIndex = atomic_fetch_add_explicit(atomicCounter, 1, memory_order_relaxed);
        
        // Check we haven't exceeded max points
        if (outputIndex < uniforms.maxPoints) {
            // Convert to camera coordinates
            float cx = float(gid.x) * uniforms.scaleX;
            float cy = float(gid.y) * uniforms.scaleY;
            sampledPoints[outputIndex] = float2(cx, cy);
        }
    }
}

