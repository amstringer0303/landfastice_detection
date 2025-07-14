# U-Net Arctic Sea Ice Classification Pipeline

This repository contains a modular deep learning pipeline for classifying sea ice features in Arctic coastal environments using Sentinel-2 RGB imagery, with a focus on **accurate landfast ice edge detection**. The model uses a 4-channel input (RGB + distance-to-coast) and predicts per-pixel semantic classes.

---

## ❄️ Target Classes

| Class ID | Label            |
|----------|------------------|
| 0        | Open Water       |
| 1        | Landfast Ice     |
| 2        | Drift Ice        |
| 3        | Transition Ice   |
| 255      | NODATA (masked)  |

---

## 🧭 Pipeline Overview

### Inputs:
- **Sentinel-2 RGB images** (`.tif`)
- **Training polygons** (`.geojson`) with `class_id` field
- **Coastline GeoJSON** (`ALASKA_63360_LN.geojson`)

---

### 1. Distance-to-Coast Raster Generation
**`distance_to_coast.py`**

- Reprojects the coastline to each image's CRS
- Rasterizes the coastline mask
- Computes a Euclidean distance transform from the coastline
- Saves per-image distance rasters (`_distance_to_coast.tif`)

---

### 2. Label Mask Rasterization
**`rasterize.py`**

- Rasterizes training polygons with `class_id` values (0–3)
- Buffers the coastline to mask inland/ambiguous zones
- Outputs NODATA=255 where excluded
- Saves full-scene masks aligned to RGB images

---

### 3. Tiling
**`tiling_wfeatures.py`**

- Matches each RGB image with its:
  - Distance-to-coast raster
  - Rasterized mask
- Splits into 320×320 tiles with stride 240
- Skips tiles with >20% NODATA
- Saves:
  - **4-band RGB + distance tiles** → `tiles/images/`
  - **1-band label masks** → `tiles/masks/`

---

### 4. Model Training
**`train.py`**

- Loads 4-channel input tiles and 1-channel categorical masks
- Uses **weighted categorical cross-entropy loss**:
  ```python
  class_weights = [3.0, 3.0, 1.0, 0.5]
