# Depth-Anything-V2 CoreML Model Setup

## Required Model
You need to download the Depth-Anything-V2 CoreML model to enable AI-powered depth estimation.

## Download Instructions

1. **Visit the Hugging Face Model Page:**
   ```
   https://huggingface.co/apple/coreml-depth-anything-v2-small
   ```

2. **Download the F16 Model:**
   - Click on "Files and versions"
   - Download: `DepthAnythingV2SmallF16.mlpackage` (49.8 MB)

3. **Install in Your Project:**
   - Place the downloaded `DepthAnythingV2SmallF16.mlpackage` folder in this `Models/` directory
   - Make sure it's added to your Xcode project target

## Alternative: Using Hugging Face CLI

If you have the Hugging Face CLI installed:

```bash
# Install CLI (if not already installed)
brew install huggingface-cli

# Download the model
huggingface-cli download \
  --local-dir SceneDepthPointCloud/Depth\ MVS/Models \
  --local-dir-use-symlinks False \
  apple/coreml-depth-anything-v2-small \
  --include "DepthAnythingV2SmallF16.mlpackage/*"
```

## Expected Performance

- **iPhone 15 Pro Max**: ~34ms inference time
- **iPhone 12 Pro Max**: ~31ms inference time  
- **Model Size**: 49.8 MB (Float16 precision)
- **Input Size**: 518x518 pixels
- **Runs on**: Neural Engine (optimized for iOS)

## Fallback Mode

If the model is not found, the app will automatically fall back to synthetic depth generation and display a warning message. The app will still function, but without AI-powered depth estimation.

## Verification

When the model loads successfully, you should see this message in the console:
```
✅ Depth-Anything-V2 model loaded successfully
MVSRenderer initialized with Depth-Anything-V2 Ready
```

If the model is missing, you'll see:
```
⚠️ Depth-Anything-V2 model not found. Using fallback depth estimation.
MVSRenderer initialized with Using Synthetic Depth
```
