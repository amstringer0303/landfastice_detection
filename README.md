# U-Net Arctic Landfast Ice Classification Pipeline

This repository contains a modular deep learning pipeline for **semantic segmentation of Arctic coastal sea ice features** using Sentinel-2 RGB imagery. The model is trained to detect and map the **landfast ice edge**, distinguishing it from open water, drift ice, and transitional zones.

The pipeline leverages a 4-channel input structure — combining RGB with a computed **distance-to-coast raster** — and uses a lightweight U-Net model trained on a curated subset of hand-labeled imagery. Once trained, the model is applied to a much larger set of scenes for full-coverage prediction.

---

## ❄️ Semantic Classes

| Class ID | Label           |
|----------|-----------------|
| 0        | Open Water      |
| 1        | Landfast Ice    |
| 2        | Drift Ice       |
| 3        | Transition Ice  |
| 255      | NODATA (masked) |

---

## 📁 Pipeline Overview

### Inputs:
- Sentinel-2 RGB imagery (`.tif`)
- GeoJSON polygons (`.geojson`) with `class_id` field for training
- Coastline geometry (`ALASKA_63360_LN.geojson`)

---

## 🛠️ Processing Steps

### 1. Distance-to-Coast Raster Generation
**Script:** `generate_all_distance_rasters.py`

- Reprojects the coastline shapefile to match each image’s CRS
- Rasterizes a binary coastline mask
- Computes a **Euclidean distance transform**
- Outputs distance rasters for all RGBs as `*_distance_to_coast.tif`

---

### 2. Label Mask Rasterization (Training Only)
**Script:** `rasterize.py`

- Rasterizes annotated training polygons into full-scene categorical masks
- Buffers inland zones to avoid ambiguous labels
- Saves aligned masks (same CRS, resolution, transform as RGBs)

---

### 3. Tiling for Training
**Script:** `tiling_wfeatures.py`

- Loads RGB + distance + mask triplets for the **5 labeled scenes**
- Tiles into 320×320 windows (stride: 240)
- Skips tiles with >20% NODATA
- Saves:
  - `tiles/images/` — 4-band RGBD training tiles
  - `tiles/masks/` — 1-band class masks
  - `tiles/distance/` — distance tiles used in stacking

---

### 4. Model Training
**Script:** `train.py`

- Loads RGBD training tiles and masks
- Applies a **U-Net model** with 4-channel input and 4 softmax output classes
- Uses **weighted categorical cross-entropy** with:
  ```python
  class_weights = [3.0, 3.0, 1.0, 0.5]
