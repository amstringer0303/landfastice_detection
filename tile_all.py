import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# === CONFIGURATION ===
tile_size = 320
stride = 240  # 25% overlap
land_threshold = 0.25  # If >25% land, skip tile

# === INPUT FILES ===
base_names = {
    "Sentinel2_Wainwright_AK_20190301": {
        "rgb": r"D:\W\2019\Sentinel2_Wainwright_AK_20190301.tif",
        "mask": r"D:\UNET\masks_rasterized\20190301_mask.tif",
        "dist": r"D:\UNET\distance_rasters\Sentinel2_Wainwright_AK_20190301_distance_to_coast.tif",
        "land": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20190301_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20220310": {
        "rgb": r"D:\W\2022\Sentinel2_Wainwright_AK_20220310.tif",
        "mask": r"D:\UNET\masks_rasterized\20220310_mask.tif",
        "dist": r"D:\UNET\distance_rasters\Sentinel2_Wainwright_AK_20220310_distance_to_coast.tif",
        "land": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20220310_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20220418": {
        "rgb": r"D:\W\2022\Sentinel2_Wainwright_AK_20220418.tif",
        "mask": r"D:\UNET\masks_rasterized\20220418_mask.tif",
        "dist": r"D:\UNET\distance_rasters\Sentinel2_Wainwright_AK_20220418_distance_to_coast.tif",
        "land": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20220418_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20230423": {
        "rgb": r"D:\W\2023\Sentinel2_Wainwright_AK_20230423.tif",
        "mask": r"D:\UNET\masks_rasterized\20230423_mask.tif",
        "dist": r"D:\UNET\distance_rasters\Sentinel2_Wainwright_AK_20230423_distance_to_coast.tif",
        "land": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20230423_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20240306": {
        "rgb": r"D:\W\2024\Sentinel2_Wainwright_AK_20240306.tif",
        "mask": r"D:\UNET\masks_rasterized\20240306_mask.tif",
        "dist": r"D:\UNET\distance_rasters\Sentinel2_Wainwright_AK_20240306_distance_to_coast.tif",
        "land": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20240306_landmask.tif"
    }
}

# === OUTPUT DIRECTORIES ===
out_base = r"D:\UNET\tiles"
out_rgb = os.path.join(out_base, "images")
out_mask = os.path.join(out_base, "masks")
out_dist = os.path.join(out_base, "distance")
os.makedirs(out_rgb, exist_ok=True)
os.makedirs(out_mask, exist_ok=True)
os.makedirs(out_dist, exist_ok=True)

# === TILING LOOP ===
for base, paths in base_names.items():
    print(f"\n🧩 Tiling: {base}")

    rgb_path = paths["rgb"]
    mask_path = paths["mask"]
    dist_path = paths["dist"]
    land_path = paths["land"]

    if not all(os.path.exists(p) for p in [rgb_path, mask_path, dist_path, land_path]):
        print(f"❌ Missing file(s) for {base} — skipping.")
        continue

    with rasterio.open(rgb_path) as rgb_src, \
         rasterio.open(mask_path) as mask_src, \
         rasterio.open(dist_path) as dist_src, \
         rasterio.open(land_path) as land_src:

        height, width = rgb_src.height, rgb_src.width
        transform = rgb_src.transform
        crs = rgb_src.crs

        count, skipped = 0, 0

        for row in tqdm(range(0, height - tile_size + 1, stride)):
            for col in range(0, width - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                transform_window = rasterio.windows.transform(window, transform)

                # Read tiles
                rgb_tile = rgb_src.read([1, 2, 3], window=window)
                mask_tile = mask_src.read(1, window=window)
                dist_tile = dist_src.read(1, window=window)
                land_tile = land_src.read(1, window=window)

                # --- FILTERS ---
                # Skip if majority land
                if np.mean(land_tile == 1) > land_threshold:
                    skipped += 1
                    continue

                # Skip if mask is only water/ignore
                if set(np.unique(mask_tile)).issubset({0, 255}):
                    skipped += 1
                    continue

                # === SAVE OUTPUTS ===
                rgb_out = os.path.join(out_rgb, f"{base}_{row}_{col}.tif")
                mask_out = os.path.join(out_mask, f"{base}_{row}_{col}.tif")
                dist_out = os.path.join(out_dist, f"{base}_{row}_{col}_distance.tif")

                with rasterio.open(rgb_out, 'w', driver='GTiff', height=tile_size, width=tile_size,
                                   count=3, dtype=rgb_tile.dtype, crs=crs, transform=transform_window) as dst:
                    dst.write(rgb_tile)

                with rasterio.open(mask_out, 'w', driver='GTiff', height=tile_size, width=tile_size,
                                   count=1, dtype=mask_tile.dtype, crs=crs, transform=transform_window) as dst:
                    dst.write(mask_tile, 1)

                with rasterio.open(dist_out, 'w', driver='GTiff', height=tile_size, width=tile_size,
                                   count=1, dtype=dist_tile.dtype, crs=crs, transform=transform_window) as dst:
                    dst.write(dist_tile, 1)

                count += 1

        print(f"✅ Done: {count} saved | 🗑️ {skipped} skipped")

