# U-Net Arctic Landfast Ice Classification Pipeline

This repository contains a modular deep learning pipeline for **semantic segmentation of Arctic coastal sea ice features** using Sentinel-2 RGB imagery. The model is trained to detect and map the **landfast ice edge**, distinguishing it from open water, drift ice, and transitional zones.

The pipeline uses a 4-channel input — combining RGB with a computed **distance-to-coast raster** — and leverages a lightweight U-Net model trained on 5 hand-labeled scenes. Once trained, the model is applied to 180+ full Sentinel-2 scenes for prediction.

---

## Semantic Classes

| Class ID | Label           |
|----------|-----------------|
| 0        | Open Water      |
| 1        | Landfast Ice    |
| 2        | Drift Ice       |
| 3        | Transition Ice  |
| 255      | NODATA (masked) |

---

## Pipeline Overview

### Inputs:
- Sentinel-2 RGB imagery (`.tif`)
- GeoJSON polygons (`.geojson`) with `class_id` field for training
- Coastline geometry (`ALASKA_63360_LN.geojson`)

---

##  Processing Steps

### 1. Distance-to-Coast Raster Generation  
**Script:** `generate_all_distance_rasters.py`

- Reprojects the coastline shapefile to match each image’s CRS
- Rasterizes a binary coastline mask
- Computes a **Euclidean distance transform**
- Saves per-image rasters as:  
  `Sentinel2_Wainwright_AK_YYYYMMDD[_version]_distance_to_coast.tif`

---

### 2. Label Mask Rasterization (Training Only)  
**Script:** `rasterize.py`

- Rasterizes hand-labeled polygons with `class_id` (0–3)
- Buffers the inland region to exclude ambiguous zones
- Applies `255 = NODATA` for excluded areas
- Outputs full-scene label masks aligned with RGB images

---

### 3. Tiling for Training  
**Script:** `tiling_wfeatures.py`

- Loads:
  - RGB image
  - Distance-to-coast raster
  - Label mask
- Tiles into 320×320 windows with 240 stride
- Skips tiles with >20% NODATA
- Saves:
  - `tiles/images/` → 4-channel RGB + distance
  - `tiles/masks/` → 1-channel label masks
  - `tiles/distance/` → distance tiles

---

### 4. Model Training  
**Script:** `train.py`

- Loads 4-band RGBD training tiles and masks
- Builds a lightweight U-Net model:
  - Input: `(320, 320, 4)`
  - Output: `(320, 320, 4)` softmax probabilities
- Uses **weighted categorical cross-entropy loss**:
  ```python
  class_weights = [3.0, 3.0, 1.0, 0.5]
  ```
- Splits dataset:
  - 80% training
  - 10% validation
  - 10% test
- Trains for **30 epochs**
- Saves model to:
  ```
  ~/Desktop/unet-4class-RGBD.keras
  ```

---

### 5. Full-Scene Inference: Tiling for Prediction

**Script:** `tile_for_inference.py`

- Scans all RGB `.tif` files in `/Volumes/toshiba/W`
- Finds matching `_distance_to_coast.tif` files
- Tiles the **entire scene** using:
  - 320×320 tile size
  - 240 stride
  - Padded edge tiles if needed
- Saves:
  - `tiles_infer/images/` → RGB tiles
  - `tiles_infer/distance/` → distance tiles

---

### 6. Model Prediction and Stitching

**Script:** `predict_and_stitch.py`

- Loads the trained model from `~/Desktop/unet-4class-RGBD.keras`
- For each tile:
  - Loads RGB + distance → 4-band input
  - Predicts class probabilities
  - Takes `argmax` to get final class mask
- Reassembles predictions into full-scene raster
- Creates color-coded overlays
- Saves to:
  - `predictions/*_stitched_overlay.png`
  - *(Optional)*: predicted raw masks or extracted boundaries


## Maintainer

Ana Stringer — [@amstringer0303](https://github.com/amstringer0303)

---
