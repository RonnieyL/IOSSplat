# Changes Summary: Model Input & Depth Photo Saving

## Changes Made

### 1. **Removed Rotation from Depth Prompt Processing**
**Previous behavior**: ARKit depth (256×192) was rotated 90° CCW → 192×256 before feeding to model

**New behavior**: ARKit depth (256×192) is used directly without rotation

**Files modified**:
- `PromptDAEngine.swift`:
  - Removed all rotation code from `makePromptArray()`
  - Changed from `CGAffineTransform(rotationAngle:)` to direct rendering
  - Added resize support if input doesn't match expected dimensions

### 2. **Updated Model Configuration**
**Previous**: 
- Model name: `PromptDA_vits_518x518_prompt192x256`
- Prompt size: 192×256 (H×W)

**New**:
- Model name: `PromptDA_vits_518x518_prompt256x192`
- Prompt size: 256×192 (H×W)

**Files modified**:
- `PromptDAEngine.swift`: Changed default parameters in `create()`
- `Renderer.swift`: Updated model name and prompt dimensions in initialization

### 3. **Added Depth Photo Saving Feature**

**New functionality**: Save ARKit depth maps as grayscale photos to Photos app

**Implementation**:
- Added `saveDepthPhoto()` public method to trigger save
- Added `saveDepthAsPhoto()` private method to convert and save
- Depth values normalized to 0-255 grayscale range
- Photos saved with sequential numbering
- Automatic permission request

**Files modified**:
- `Renderer.swift`:
  - Imported `Photos` framework
  - Added `depthPhotoCounter` and `shouldSaveNextDepth` properties
  - Added depth saving logic in `updateLiDARDepthTextures()`
  - Created `saveDepthAsPhoto()` method with normalization
- `Info.plist`:
  - Added `NSPhotoLibraryAddUsageDescription` key

## How to Use

### **Trigger Depth Photo Save**
Call from your UI (e.g., MainController):
```swift
@IBAction func saveDepthPhotoButtonTapped(_ sender: UIButton) {
    renderer.saveDepthPhoto()
}
```

The next depth frame will be:
1. Normalized to grayscale (0-255)
2. Converted to UIImage
3. Saved to Photos app
4. Logged with: `✅ Depth photo #N saved to Photos (256×192)`

### **Expected Console Output**

When depth photo is saved:
```
📸 Saving ARKit depth map as photo...
   • Depth range: 0.234m - 4.567m
   ✅ Depth photo #1 saved to Photos (256×192)
```

## Model Requirements

### **Old Model** (no longer used):
- Name: `PromptDA_vits_518x518_prompt192x256.mlpackage`
- Inputs:
  - `colorImage`: 518×518 RGB
  - `promptDepth`: [1, 1, 192, 256] (rotated ARKit depth)

### **New Model** (current):
- Name: `PromptDA_vits_518x518_prompt256x192.mlpackage`
- Inputs:
  - `colorImage`: 518×518 RGB
  - `promptDepth`: [1, 1, 256, 192] (direct ARKit depth, no rotation)

**Important**: Make sure your CoreML model expects 256×192 prompt depth!

## Depth Processing Pipeline

### **Before (with rotation)**:
```
ARKit Smoothed Depth (256×192)
    ↓
Rotate 90° CCW
    ↓
Depth Prompt (192×256)
    ↓
CoreML Model
```

### **After (no rotation)**:
```
ARKit Smoothed Depth (256×192)
    ↓
Direct Copy (or resize if needed)
    ↓
Depth Prompt (256×192)
    ↓
CoreML Model
```

## Depth Photo Format

- **Resolution**: 256×192 (matches ARKit depth)
- **Format**: 8-bit grayscale PNG
- **Range**: Normalized from actual depth range (e.g., 0.2m - 5.0m) to 0-255
- **Location**: iOS Photos app
- **Naming**: Sequential counter (tracked in `depthPhotoCounter`)

## Permissions

The app will automatically request Photos permission when you call `saveDepthPhoto()` for the first time.

**Info.plist entry**:
```xml
<key>NSPhotoLibraryAddUsageDescription</key>
<string>Save depth map images to Photos.</string>
```

## Debug Output Changes

### **Model Initialization**:
```
🚀 PromptDAEngine Initialization Starting...
   • Model name: PromptDA_vits_518x518_prompt256x192
   • RGB input size: 518×518
   • Prompt depth size: 256×192
   • Note: Matches ARKit smoothed depth (256×192) directly, no rotation
```

### **Prompt Processing**:
```
      → makePromptArray: input 256×192, using directly (no rotation)
      → Rendering to temp buffer: 256×192
      → Prompt depth stats: min=0.234m, max=4.567m, valid=48234/49152
```

## Testing Checklist

- [ ] Verify model name matches: `PromptDA_vits_518x518_prompt256x192.mlpackage`
- [ ] Check model input shape: `[1, 1, 256, 192]` for promptDepth
- [ ] Add model file to Xcode project target
- [ ] Grant Photos permission when prompted
- [ ] Test depth photo saving in LiDAR mode
- [ ] Verify saved photos appear in Photos app
- [ ] Check console for depth range and save confirmation

## Summary

✅ **Rotation removed**: ARKit depth used directly (256×192)  
✅ **Model updated**: Now expects 256×192 prompt (not 192×256)  
✅ **Photo saving added**: Depth maps saved as grayscale images  
✅ **Permissions added**: Photos library access configured  
✅ **Debug enhanced**: Clear logging for depth processing and saving
