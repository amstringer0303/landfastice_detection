import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# === CONFIGURATION ===
tile_size = 320
stride = 240  # 25% overlap

# Input paths
rgb_dir = r"D:\W"
mask_dir = r"D:\UNET\masks_rasterized"
dist_dir = r"D:\UNET\distance_rasters"

# Output paths
out_base = r"D:\UNET\tiles"
out_rgb = os.path.join(out_base, "images")
out_mask = os.path.join(out_base, "masks")
out_dist = os.path.join(out_base, "distance")

os.makedirs(out_rgb, exist_ok=True)
os.makedirs(out_mask, exist_ok=True)
os.makedirs(out_dist, exist_ok=True)

# Get filenames (shared across all 3 inputs)
base_names = [
    "Sentinel2_Wainwright_AK_20190301",
    "Sentinel2_Wainwright_AK_20220310",
    "Sentinel2_Wainwright_AK_20220418",
    "Sentinel2_Wainwright_AK_20230423",
    "Sentinel2_Wainwright_AK_20240306"
]

for base in base_names:
    print(f"\n🧩 Tiling scene: {base}")

    rgb_path = os.path.join(rgb_dir, base.replace("AK_", "AK_") + ".tif")
    mask_path = os.path.join(mask_dir, f"{base.split('_')[-1]}_mask.tif")
    dist_path = os.path.join(dist_dir, f"{base}_distance_to_coast.tif")

    if not (os.path.exists(rgb_path) and os.path.exists(mask_path) and os.path.exists(dist_path)):
        print(f"⚠️ Missing file(s) for {base} — skipping.")
        continue

    with rasterio.open(rgb_path) as rgb_src, \
         rasterio.open(mask_path) as mask_src, \
         rasterio.open(dist_path) as dist_src:

        height, width = rgb_src.height, rgb_src.width
        transform = rgb_src.transform
        crs = rgb_src.crs

        count = 0
        for row in tqdm(range(0, height - tile_size + 1, stride)):
            for col in range(0, width - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                transform_window = rasterio.windows.transform(window, transform)

                # RGB
                rgb_tile = rgb_src.read([1, 2, 3], window=window)
                rgb_out = os.path.join(out_rgb, f"{base}_{row}_{col}.tif")
                with rasterio.open(rgb_out, 'w', driver='GTiff', height=tile_size, width=tile_size,
                                   count=3, dtype=rgb_tile.dtype, crs=crs, transform=transform_window) as dst:
                    dst.write(rgb_tile)

                # Mask
                mask_tile = mask_src.read(1, window=window)
                mask_out = os.path.join(out_mask, f"{base}_{row}_{col}.tif")
                with rasterio.open(mask_out, 'w', driver='GTiff', height=tile_size, width=tile_size,
                                   count=1, dtype=mask_tile.dtype, crs=crs, transform=transform_window) as dst:
                    dst.write(mask_tile, 1)

                # Distance
                dist_tile = dist_src.read(1, window=window)
                dist_out = os.path.join(out_dist, f"{base}_{row}_{col}_distance.tif")
                with rasterio.open(dist_out, 'w', driver='GTiff', height=tile_size, width=tile_size,
                                   count=1, dtype=dist_tile.dtype, crs=crs, transform=transform_window) as dst:
                    dst.write(dist_tile, 1)

                count += 1

        print(f"✅ Saved {count} tiles for {base}")
