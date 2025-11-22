#include <metal_stdlib>
using namespace metal;

// Constants for radix sort
constant uint RADIX_BITS = 8;
constant uint RADIX_SIZE = 256;  // 2^8
constant uint RADIX_MASK = 255;  // 0xFF
constant uint NUM_PASSES = 4;    // For 32-bit floats

// Structure for sorting: index and depth key
struct SplatIndexAndDepth {
    uint index;
    float depth;
};

// Convert float to sortable uint (handles negative floats correctly)
inline uint floatToSortableUint(float f) {
    uint u = as_type<uint>(f);
    // Flip all bits if negative, otherwise flip only sign bit
    uint mask = -int(u >> 31) | 0x80000000;
    return u ^ mask;
}

// Convert back from sortable uint to float
inline float sortableUintToFloat(uint u) {
    uint mask = ((u >> 31) - 1) | 0x80000000;
    return as_type<float>(u ^ mask);
}

// Extract radix digit from sortable key
inline uint getRadixDigit(uint key, uint pass) {
    uint shift = pass * RADIX_BITS;
    return (key >> shift) & RADIX_MASK;
}

// KERNEL 1: Calculate depths for all splats
// This replaces the CPU depth calculation
kernel void calculateDepths(
    constant float3* positions [[buffer(0)]],
    constant float3& cameraPosition [[buffer(1)]],
    constant bool& sortByDistance [[buffer(2)]],
    constant float3& cameraForward [[buffer(3)]],
    device SplatIndexAndDepth* output [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    float3 splatPosition = positions[gid];

    float depth;
    if (sortByDistance) {
        float3 diff = splatPosition - cameraPosition;
        depth = dot(diff, diff);  // Squared distance
    } else {
        depth = dot(splatPosition, cameraForward);
    }

    output[gid].index = gid;
    output[gid].depth = depth;
}

// KERNEL 2: Local histogram (per threadgroup)
// Each threadgroup computes histogram for its portion of data
kernel void localHistogram(
    device SplatIndexAndDepth* input [[buffer(0)]],
    device atomic_uint* globalHistogram [[buffer(1)]],
    constant uint& numElements [[buffer(2)]],
    constant uint& pass [[buffer(3)]],
    threadgroup atomic_uint* localHist [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]
) {
    // Initialize local histogram
    for (uint i = lid; i < RADIX_SIZE; i += groupSize) {
        atomic_store_explicit(&localHist[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count local occurrences
    if (gid < numElements) {
        uint key = floatToSortableUint(input[gid].depth);
        uint digit = getRadixDigit(key, pass);
        atomic_fetch_add_explicit(&localHist[digit], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Accumulate to global histogram
    for (uint i = lid; i < RADIX_SIZE; i += groupSize) {
        uint count = atomic_load_explicit(&localHist[i], memory_order_relaxed);
        if (count > 0) {
            atomic_fetch_add_explicit(&globalHistogram[i], count, memory_order_relaxed);
        }
    }
}

// KERNEL 3: Prefix sum (scan) using Blelloch algorithm
// Computes cumulative offsets from histogram
kernel void prefixSum(
    device uint* histogram [[buffer(0)]],
    device uint* prefixSums [[buffer(1)]],
    threadgroup uint* temp [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]
) {
    // Load input into shared memory
    if (lid < RADIX_SIZE) {
        temp[lid] = histogram[lid];
    } else {
        temp[lid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    uint offset = 1;
    for (uint d = groupSize >> 1; d > 0; d >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai = offset * (2 * lid + 1) - 1;
            uint bi = offset * (2 * lid + 2) - 1;
            if (bi < RADIX_SIZE) {
                temp[bi] += temp[ai];
            }
        }
        offset *= 2;
    }

    // Clear last element
    if (lid == 0) {
        temp[RADIX_SIZE - 1] = 0;
    }

    // Down-sweep phase
    for (uint d = 1; d < groupSize; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai = offset * (2 * lid + 1) - 1;
            uint bi = offset * (2 * lid + 2) - 1;
            if (bi < RADIX_SIZE) {
                uint t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    if (lid < RADIX_SIZE) {
        prefixSums[lid] = temp[lid];
    }
}

// KERNEL 4: Reorder elements based on radix digit
// Scatters elements to their sorted positions
kernel void reorder(
    device SplatIndexAndDepth* input [[buffer(0)]],
    device SplatIndexAndDepth* output [[buffer(1)]],
    device uint* prefixSums [[buffer(2)]],
    device atomic_uint* localOffsets [[buffer(3)]],
    constant uint& numElements [[buffer(4)]],
    constant uint& pass [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numElements) return;

    SplatIndexAndDepth element = input[gid];
    uint key = floatToSortableUint(element.depth);
    uint digit = getRadixDigit(key, pass);

    // Get base offset for this digit
    uint baseOffset = prefixSums[digit];

    // Atomically get local offset within this digit's range
    uint localOffset = atomic_fetch_add_explicit(&localOffsets[digit], 1, memory_order_relaxed);

    // Write to output position
    uint outputPos = baseOffset + localOffset;
    if (outputPos < numElements) {
        output[outputPos] = element;
    }
}

// KERNEL 5: Final reorder - map sorted indices back to splat data
// Copies splats from old buffer to new buffer using sorted indices
kernel void reorderSplats(
    device void* inputSplats [[buffer(0)]],
    device void* outputSplats [[buffer(1)]],
    device uint* sortedIndices [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    constant uint& splatStride [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numElements) return;

    uint originalIndex = sortedIndices[gid];

    // Copy splat data (raw memory copy)
    device char* src = (device char*)inputSplats + (originalIndex * splatStride);
    device char* dst = (device char*)outputSplats + (gid * splatStride);

    for (uint i = 0; i < splatStride; i++) {
        dst[i] = src[i];
    }
}
