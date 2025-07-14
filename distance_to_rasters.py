import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# === CONFIGURATION ===
input_dir = os.path.expanduser("~/Desktop/distance_rasters/")
output_dir = "/Volumes/toshiba/UNET/tiles/distance/"
tile_size = 320
stride = 240

os.makedirs(output_dir, exist_ok=True)

# === GET FILES ===
distance_files = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith("_distance_to_coast.tif")
])

print(f"🌊 Found {len(distance_files)} distance rasters.")

# === TILING LOOP ===
for dist_path in distance_files:
    # Preserve full base name, including any "_fixed"
    base = os.path.basename(dist_path).replace("_distance_to_coast.tif", "")
    print(f"\n🧩 Tiling distance raster: {base}")

    with rasterio.open(dist_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

        count = 0
        for row in tqdm(range(0, height - tile_size + 1, stride)):
            for col in range(0, width - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)
                transform_window = rasterio.windows.transform(window, transform)
                tile = src.read(1, window=window)

                # Save tile with base name intact
                out_path = os.path.join(output_dir, f"{base}_{row}_{col}_distance.tif")
                with rasterio.open(
                    out_path, 'w',
                    driver='GTiff',
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype=tile.dtype,
                    crs=crs,
                    transform=transform_window
                ) as dst:
                    dst.write(tile, 1)

                count += 1

        print(f"✅ Saved {count} distance tiles for {base}")
