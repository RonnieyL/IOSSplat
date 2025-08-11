# Build Instructions for MVS Fix

## Steps to Fix the "Unknown class MVSController" Error:

### 1. Clean Build (Essential)
```
Product → Clean Build Folder (Cmd+Shift+K)
```

### 2. Check File Target Membership
In Xcode:
1. Select `MVSController.swift` in Project Navigator
2. In the right panel, check "Target Membership"
3. Ensure your app target is checked ✅
4. Do the same for all MVS files:
   - MVSRenderer.swift
   - DepthAnythingProcessor.swift
   - MVSParticle.swift
   - MVSMetalBuffer.swift
   - MVSError.swift
   - MVSShaderTypes.h
   - MVSShaders.metal

### 3. Build Project
```
Product → Build (Cmd+B)
```

### 4. If Still Failing
Open the storyboard in Xcode:
1. Select the MVS Controller scene
2. In Identity Inspector (right panel):
   - Class: MVSController
   - Module: (leave blank or set to your app name)
   - Click outside to refresh

### 5. Alternative: Recreate the Segue
If the above doesn't work:
1. Delete the existing segue from Depth MVS button
2. Control+drag from button to MVS Controller
3. Choose "Show" segue
4. Set identifier to "showDepthMVS"

### 6. Check Bundle Identifier
Make sure your app's bundle identifier matches what's expected.
