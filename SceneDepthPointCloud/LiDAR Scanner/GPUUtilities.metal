#include <metal_stdlib>
using namespace metal;

// MARK: - Prefix Sum (Exclusive Scan)

/**
 Exclusive prefix sum for Int32 arrays

 Simple single-threaded implementation for prototype
 TODO: Replace with parallel scan (Blelloch algorithm) for better performance
 */
kernel void exclusivePrefixSumInt32(
    constant int32_t* input [[buffer(0)]],
    device int32_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single-threaded for now

    int32_t sum = 0;
    for (uint i = 0; i < count; i++) {
        output[i] = sum;
        sum += input[i];
    }
}

/**
 Parallel prefix sum using Blelloch scan algorithm
 More efficient for large arrays (work-efficient O(n) instead of O(n log n))
 */
kernel void parallelPrefixSumInt32(
    device int32_t* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    threadgroup int32_t* temp [[threadgroup(0)]],
    uint lid [[thread_position_in_threadgroup]],
    uint groupSize [[threads_per_threadgroup]]
) {
    // Load input into shared memory
    if (lid < n) {
        temp[lid] = data[lid];
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
            if (bi < n) {
                temp[bi] += temp[ai];
            }
        }
        offset *= 2;
    }

    // Clear last element (for exclusive scan)
    if (lid == 0) {
        temp[n - 1] = 0;
    }

    // Down-sweep phase
    for (uint d = 1; d < groupSize; d *= 2) {
        offset >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < d) {
            uint ai = offset * (2 * lid + 1) - 1;
            uint bi = offset * (2 * lid + 2) - 1;
            if (bi < n) {
                int32_t t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write results
    if (lid < n) {
        data[lid] = temp[lid];
    }
}

// MARK: - Gather Operations

/**
 Gather (reorder) Int32 array based on indices

 output[i] = source[indices[i]]
 */
kernel void gatherInt32(
    constant uint32_t* indices [[buffer(0)]],
    constant int32_t* source [[buffer(1)]],
    device int32_t* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint srcIndex = indices[gid];
    destination[gid] = source[srcIndex];
}

/**
 Gather (reorder) Int64 array based on indices
 */
kernel void gatherInt64(
    constant uint32_t* indices [[buffer(0)]],
    constant int64_t* source [[buffer(1)]],
    device int64_t* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint srcIndex = indices[gid];
    destination[gid] = source[srcIndex];
}

/**
 Gather (reorder) Float array based on indices
 */
kernel void gatherFloat(
    constant uint32_t* indices [[buffer(0)]],
    constant float* source [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint srcIndex = indices[gid];
    destination[gid] = source[srcIndex];
}

/**
 Gather packed float2 arrays (e.g., xys in OpenSplat)
 Metal pads float2 arrays, so use manual indexing
 */
kernel void gatherPackedFloat2(
    constant uint32_t* indices [[buffer(0)]],
    constant float* source [[buffer(1)]],      // Packed as [x0, y0, x1, y1, ...]
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint srcIndex = indices[gid];
    destination[2 * gid + 0] = source[2 * srcIndex + 0];
    destination[2 * gid + 1] = source[2 * srcIndex + 1];
}

/**
 Gather packed float3 arrays (e.g., colors, conics)
 */
kernel void gatherPackedFloat3(
    constant uint32_t* indices [[buffer(0)]],
    constant float* source [[buffer(1)]],      // Packed as [x0, y0, z0, x1, y1, z1, ...]
    device float* destination [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint srcIndex = indices[gid];
    destination[3 * gid + 0] = source[3 * srcIndex + 0];
    destination[3 * gid + 1] = source[3 * srcIndex + 1];
    destination[3 * gid + 2] = source[3 * srcIndex + 2];
}

// MARK: - Helper: Initialize Indices

/**
 Initialize index array to [0, 1, 2, ..., count-1]
 Useful for creating initial indices before sorting
 */
kernel void initializeIndices(
    device uint32_t* indices [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    indices[gid] = gid;
}
