# OpenSplat Rendering Pipeline Reference

This document describes the exact OpenSplat rendering pipeline with pure C++/Metal equivalent code.
Use this to verify your Swift implementation matches OpenSplat exactly.

---

## INPUT DATA

```
Camera Intrinsics (from COLMAP/dataset):
  - fx, fy: focal lengths in PIXELS (e.g., fx=1200, fy=1200 for a 1920x1080 image)
  - cx, cy: principal point in PIXELS (usually image_width/2, image_height/2)
  - width, height: image dimensions in pixels

Camera Extrinsics:
  - camToWorld: 4x4 matrix transforming from camera space to world space
    (This is the INVERSE of what ARKit gives you - ARKit gives worldToCamera)

Gaussian Data:
  - means: Nx3 float, 3D positions in world space
  - scales: Nx3 float, log-scale values (actual scale = exp(scales))
  - quats: Nx4 float, quaternions (w, x, y, z) - NOT normalized in storage
  - opacities: Nx1 float, logit values (actual opacity = sigmoid(opacities))
  - colors: Nx3 float, RGB in [0,1] range (or SH coefficients)
```

---

## STEP 1: BUILD VIEW MATRIX

OpenSplat has `camToWorld` (camera-to-world transform).
ARKit gives you `viewMatrix` which IS the world-to-camera transform already.

```cpp
// OpenSplat (model.cpp lines 93-107):
// They START with camToWorld and must invert it

// Extract rotation and translation from camToWorld
float3x3 R = camToWorld[0:3, 0:3];  // 3x3 rotation
float3 T = camToWorld[0:3, 3];       // 3x1 translation

// CRITICAL: Flip Y and Z to match gsplat coordinate convention
// gsplat expects: camera looks down +Z, Y points down
// Standard: camera looks down -Z, Y points up
R = R * diag(1, -1, -1);  // Post-multiply to flip axes

// Invert to get worldToCamera (view matrix)
float3x3 Rinv = transpose(R);        // For orthonormal matrix, inverse = transpose
float3 Tinv = -Rinv * T;

// Build 4x4 view matrix (world-to-camera)
float4x4 viewMat = identity();
viewMat[0:3, 0:3] = Rinv;
viewMat[0:3, 3] = Tinv;
```

### For ARKit (Swift equivalent):

```swift
// ARKit gives us viewMatrix directly (world-to-camera)
// But we still need to apply the coordinate flip

// The flip is applied to the VIEW MATRIX, not the camera transform
// Since ARKit's viewMatrix = inverse(camToWorld), we apply flip differently:
// 
// OpenSplat does: viewMat = inverse(camToWorld * flipYZ)
//                         = inverse(flipYZ) * inverse(camToWorld)
//                         = flipYZ * viewMatrix  (since flipYZ is self-inverse)
//
// BUT they do: R = R * diag(1,-1,-1) BEFORE inverting
// So the equivalent for ARKit is: viewMatrix * flipYZ

let flipYZ = matrix_float4x4(columns: (
    SIMD4<Float>(1, 0, 0, 0),
    SIMD4<Float>(0, -1, 0, 0),
    SIMD4<Float>(0, 0, -1, 0),
    SIMD4<Float>(0, 0, 0, 1)
))
let adjustedViewMatrix = viewMatrix * flipYZ
```

---

## STEP 2: BUILD PROJECTION MATRIX

OpenSplat builds its own OpenGL-style projection matrix.

```cpp
// OpenSplat (model.cpp lines 35-47 and 110-113):

// Compute FOV from focal lengths
float fovX = 2.0f * atan(width / (2.0f * fx));
float fovY = 2.0f * atan(height / (2.0f * fy));

// Build OpenGL perspective projection matrix
float zNear = 0.001f;
float zFar = 1000.0f;

float t = zNear * tan(0.5f * fovY);  // top
float b = -t;                          // bottom
float r = zNear * tan(0.5f * fovX);  // right  
float l = -r;                          // left

// The projection matrix (ROW-MAJOR as stored in PyTorch):
// Row 0: [2n/(r-l),    0,          (r+l)/(r-l),  0        ]
// Row 1: [0,           2n/(t-b),   (t+b)/(t-b),  0        ]
// Row 2: [0,           0,          (f+n)/(f-n),  -fn/(f-n)]
// Row 3: [0,           0,          1,            0        ]
//
// Note: Row 3 = [0,0,1,0] means w_clip = z_view (OpenGL convention)
// This is DIFFERENT from Metal's default projection!

float4x4 projMat;
projMat[0] = {2*zNear/(r-l), 0,             (r+l)/(r-l),           0};
projMat[1] = {0,             2*zNear/(t-b), (t+b)/(t-b),           0};
projMat[2] = {0,             0,             (zFar+zNear)/(zFar-zNear), -zFar*zNear/(zFar-zNear)};
projMat[3] = {0,             0,             1,                      0};
```

### Swift equivalent:

```swift
// Extract focal lengths from ARKit projection matrix
// ARKit's projection: P[0][0] = 2*fx/width (for symmetric frustum)
let fx = abs(projectionMatrix.columns.0.x) * Float(width) / 2.0
let fy = abs(projectionMatrix.columns.1.y) * Float(height) / 2.0
let cx = Float(width) / 2.0
let cy = Float(height) / 2.0

// Compute FOV
let fovX = 2.0 * atan(Float(width) / (2.0 * fx))
let fovY = 2.0 * atan(Float(height) / (2.0 * fy))

// Build OpenGL-style projection matrix
let zNear: Float = 0.001
let zFar: Float = 1000.0
let t = zNear * tan(0.5 * fovY)
let b = -t
let r = zNear * tan(0.5 * fovX)
let l = -r

// Swift uses COLUMN-MAJOR, so we specify columns
// Column 0 = [row0[0], row1[0], row2[0], row3[0]]
let gsplatProjMatrix = matrix_float4x4(columns: (
    SIMD4<Float>(2*zNear/(r-l), 0, 0, 0),
    SIMD4<Float>(0, 2*zNear/(t-b), 0, 0),
    SIMD4<Float>((r+l)/(r-l), (t+b)/(t-b), (zFar+zNear)/(zFar-zNear), 1),
    SIMD4<Float>(0, 0, -zFar*zNear/(zFar-zNear), 0)
))
```

---

## STEP 3: COMBINE MATRICES FOR SHADER

**CRITICAL**: OpenSplat passes `projMat * viewMat` to the projection shader!

```cpp
// OpenSplat (model.cpp line 153):
torch::matmul(projMat, viewMat)  // This is passed as "projmat" to shader
```

### Swift equivalent:

```swift
let combinedMatrix = gsplatProjMatrix * adjustedViewMatrix
```

---

## STEP 4: CONVERT TO ROW-MAJOR FOR SHADER

The Metal shader (`transform_4x4`) expects row-major layout:

```metal
// gsplat_metal.metal lines 100-108:
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],   // row 0
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],   // row 1
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11], // row 2
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15], // row 3
    };
    return out;
}
```

Swift's `matrix_float4x4` is column-major. To convert to row-major, transpose:

```swift
var viewMat = adjustedViewMatrix.transpose
var projMat = combinedMatrix.transpose
```

---

## STEP 5: PROJECT GAUSSIANS KERNEL

```metal
// gsplat_metal.metal project_gaussians_forward_kernel:

kernel void project_gaussians_forward_kernel(
    constant int& num_points,
    constant float* means3d,      // Nx3 packed
    constant float* scales,       // Nx3 packed  
    constant float& glob_scale,   // usually 1.0
    constant float* quats,        // Nx4 packed
    constant float* viewmat,      // 4x4 row-major
    constant float* projmat,      // 4x4 row-major (THIS IS projMat * viewMat!)
    constant float4& intrins,     // (fx, fy, cx, cy)
    constant uint2& img_size,     // (width, height)
    constant uint3& tile_bounds,  // (tiles_x, tiles_y, 1)
    constant float& clip_thresh,  // usually 0.01
    device float* covs3d,         // output: Nx6
    device float* xys,            // output: Nx2 screen positions
    device float* depths,         // output: Nx1 view-space Z
    device int* radii,            // output: Nx1 screen radius
    device float* conics,         // output: Nx3 inverse 2D covariance
    device int32_t* num_tiles_hit // output: Nx1 tile count
) {
    uint idx = gp.x;
    if (idx >= num_points) return;
    
    // Initialize outputs
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;
    
    // Read world position
    float3 p_world = {means3d[idx*3], means3d[idx*3+1], means3d[idx*3+2]};
    
    // Transform to view space using viewmat
    float3 p_view = transform_4x3(viewmat, p_world);
    
    // Clip if behind near plane
    // CRITICAL: gsplat expects p_view.z > 0 for visible points!
    if (p_view.z <= clip_thresh) return;
    
    // Compute 3D covariance from scale and quaternion
    float3 scale = {scales[idx*3], scales[idx*3+1], scales[idx*3+2]};
    float4 quat = {quats[idx*4], quats[idx*4+1], quats[idx*4+2], quats[idx*4+3]};
    scale_rot_to_cov3d(scale, glob_scale, quat, &covs3d[idx*6]);
    
    // Project 3D covariance to 2D using EWA approximation
    float fx = intrins.x, fy = intrins.y, cx = intrins.z, cy = intrins.w;
    float tan_fovx = 0.5 * img_size.x / fx;
    float tan_fovy = 0.5 * img_size.y / fy;
    float3 cov2d = project_cov3d_ewa(p_world, &covs3d[idx*6], viewmat, fx, fy, tan_fovx, tan_fovy);
    
    // Compute conic (inverse of 2D covariance) and radius
    float3 conic;
    float radius;
    if (!compute_cov2d_bounds(cov2d, conic, radius)) return;
    conics[idx*3] = conic.x;
    conics[idx*3+1] = conic.y;
    conics[idx*3+2] = conic.z;
    
    // PROJECT TO SCREEN PIXELS using combined projmat (projMat * viewMat)
    float2 center = project_pix(projmat, p_world, img_size, {cx, cy});
    
    // Compute tile bounds
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) return;
    
    // Write outputs
    num_tiles_hit[idx] = tile_area;
    depths[idx] = p_view.z;
    radii[idx] = (int)radius;
    xys[idx*2] = center.x;
    xys[idx*2+1] = center.y;
}
```

### The project_pix function:

```metal
inline float2 project_pix(constant float *mat, float3 p, uint2 img_size, float2 pp) {
    // mat is the COMBINED projMat * viewMat, row-major
    // pp is principal point (cx, cy)
    
    // Transform world point to clip space
    float4 p_hom = transform_4x4(mat, p);
    
    // Perspective divide to get NDC
    float rw = 1.0 / (p_hom.w + 1e-6);
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    // Now p_proj.x and p_proj.y are in NDC [-1, +1]
    
    // Convert NDC to pixel coordinates
    return {
        ndc2pix(p_proj.x, img_size.x, pp.x),
        ndc2pix(p_proj.y, img_size.y, pp.y)
    };
}

inline float ndc2pix(float x, float W, float cx) {
    // x is in NDC [-1, +1]
    // W is image width (or height)
    // cx is principal point (usually W/2)
    return 0.5 * W * x + cx - 0.5;
}

// For NDC in [-1, +1] and cx = W/2:
// ndc2pix(-1, W, W/2) = -W/2 + W/2 - 0.5 = -0.5 ≈ 0 (left edge)
// ndc2pix(+1, W, W/2) = +W/2 + W/2 - 0.5 = W - 0.5 ≈ W (right edge)
```

---

## STEP 6: BIN AND SORT GAUSSIANS

```cpp
// After projection, for each Gaussian we know:
// - xys[i]: 2D screen position
// - depths[i]: view-space Z depth  
// - radii[i]: screen-space radius
// - num_tiles_hit[i]: how many tiles it overlaps

// 6.1: Cumulative sum of tiles hit
int cumTilesHit[N];
cumTilesHit[0] = 0;
for (int i = 1; i < N; i++) {
    cumTilesHit[i] = cumTilesHit[i-1] + num_tiles_hit[i-1];
}
int totalIntersects = cumTilesHit[N-1] + num_tiles_hit[N-1];

// 6.2: Map each Gaussian to tile intersections
// For each Gaussian, for each tile it overlaps:
//   isectIds[k] = (tile_id << 32) | depth_bits
//   gaussianIds[k] = gaussian_index
map_gaussian_to_intersects(xys, depths, radii, cumTilesHit, tile_bounds,
                           isectIds, gaussianIds);

// 6.3: Sort by isectIds (sorts by tile first, then by depth within tile)
sort(isectIds, gaussianIds);  // Sort both arrays together by isectIds

// 6.4: Compute tile bin edges
// tileBins[tile_id] = (start_index, end_index) in sorted arrays
get_tile_bin_edges(totalIntersects, isectIds, tileBins);
```

---

## STEP 7: RASTERIZE

```metal
kernel void nd_rasterize_forward_kernel(
    constant uint3& tile_bounds,
    constant uint3& img_size,
    constant uint& channels,           // 3 for RGB
    constant int32_t* gaussian_ids_sorted,
    constant int* tile_bins,           // (start, end) per tile
    constant float* xys,               // 2D screen positions
    constant float* conics,            // inverse 2D covariance
    constant float* colors,            // RGB colors
    constant float* opacities,
    device float* final_Ts,
    device int* final_index,
    device float* out_img,
    constant float* background,
    constant uint2& blockDim,          // (16, 16)
    uint2 blockIdx [[threadgroup_position_in_grid]],
    uint2 threadIdx [[thread_position_in_threadgroup]]
) {
    // Compute pixel position
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column
    float px = (float)j;
    float py = (float)i;
    int pix_id = i * img_size.x + j;
    
    if (i >= img_size.y || j >= img_size.x) return;
    
    // Get tile ID and gaussian range for this tile
    int tile_id = blockIdx.y * tile_bounds.x + blockIdx.x;
    int2 range = {tile_bins[tile_id*2], tile_bins[tile_id*2+1]};
    
    float T = 1.0;  // Transmittance
    
    // Iterate over gaussians in this tile (front to back by depth)
    for (int idx = range.x; idx < range.y; idx++) {
        int g = gaussian_ids_sorted[idx];
        
        // Get gaussian 2D position and conic
        float2 center = {xys[g*2], xys[g*2+1]};
        float3 conic = {conics[g*3], conics[g*3+1], conics[g*3+2]};
        
        // Distance from pixel to gaussian center
        float2 delta = {center.x - px, center.y - py};
        
        // Mahalanobis distance (Gaussian falloff)
        // sigma = 0.5 * delta^T * conic * delta
        float sigma = 0.5 * (conic.x * delta.x * delta.x + 
                             conic.z * delta.y * delta.y) +
                      conic.y * delta.x * delta.y;
        
        if (sigma < 0) continue;
        
        // Compute alpha
        float opac = opacities[g];
        float alpha = min(0.999, opac * exp(-sigma));
        
        if (alpha < 1.0/255.0) continue;
        
        // Alpha compositing (front-to-back)
        float vis = alpha * T;
        for (int c = 0; c < channels; c++) {
            out_img[pix_id * channels + c] += colors[g * channels + c] * vis;
        }
        
        T *= (1 - alpha);
        
        if (T <= 1e-4) break;  // Early termination
    }
    
    // Add background
    for (int c = 0; c < channels; c++) {
        out_img[pix_id * channels + c] += T * background[c];
    }
    
    final_Ts[pix_id] = T;
}
```

---

## CRITICAL DIFFERENCES TO CHECK IN YOUR IMPLEMENTATION

### 1. Coordinate Flip Order
OpenSplat: `R = R * diag(1, -1, -1)` applied BEFORE inverting
Your Swift: `viewMatrix * flipYZ` - this should be equivalent since ARKit already inverted

### 2. Combined Projection Matrix
OpenSplat: Passes `projMat * viewMat` to shader as "projmat"
Your Swift: Must do `gsplatProjMatrix * adjustedViewMatrix`

### 3. Row-Major Conversion
OpenSplat: PyTorch tensors are row-major by default
Your Swift: Must call `.transpose` on matrices before passing to shader

### 4. Principal Point
OpenSplat: Uses actual `cx, cy` from camera calibration
Your Swift: Using `width/2, height/2` - this is correct for ARKit which centers the principal point

### 5. Intrinsics for Covariance Projection
The intrinsics `(fx, fy, cx, cy)` are used in TWO places:
- In `project_pix`: only `cx, cy` used (as principal point offset in ndc2pix)
- In `project_cov3d_ewa`: `fx, fy` used for Jacobian, plus tan_fov computed from them

### 6. Output Image Buffer
Must be ZEROED before rasterization! The kernel uses `+=` accumulation.

---

## DEBUGGING CHECKLIST

1. **Print fx, fy values**: Should be ~1000-2000 for a phone camera
2. **Print fovX, fovY**: Should be ~60-90 degrees
3. **Print sample world positions**: Make sure points are in reasonable range (a few meters)
4. **Print sample projected XY**: Should span [0, width] x [0, height]
5. **Check depths**: Should be positive for visible points (after coordinate flip)

If XY values are clustered in a small region:
- Check that combined matrix is `projMat * viewMat`, not just `projMat`
- Check row-major conversion (transpose)
- Check coordinate flip is applied correctly
