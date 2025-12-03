#include <metal_stdlib>
using namespace metal;

kernel void bitonic_sort_kernel(
    device uint64_t* keys [[buffer(0)]],
    device int32_t* values [[buffer(1)]],
    constant uint& j [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint ixj = id ^ j;
    
    if (ixj > id) {
        bool ascending = (id & k) == 0;
        uint64_t key_i = keys[id];
        uint64_t key_j = keys[ixj];
        
        if ((ascending && key_i > key_j) || (!ascending && key_i < key_j)) {
             keys[id] = key_j;
             keys[ixj] = key_i;
             
             int32_t val_i = values[id];
             int32_t val_j = values[ixj];
             values[id] = val_j;
             values[ixj] = val_i;
        }
    }
}

kernel void bitonic_pad_kernel(
    device uint64_t* keys [[buffer(0)]],
    constant uint& start_index [[buffer(1)]],
    constant uint& end_index [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint idx = start_index + id;
    if (idx < end_index) {
        keys[idx] = 0xFFFFFFFFFFFFFFFFUL; // UINT64_MAX
    }
}
