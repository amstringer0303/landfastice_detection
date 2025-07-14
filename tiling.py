import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import glob
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

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

# === LOADER FUNCTION FOR U-NET INPUT ===
def load_geotiff_image_with_distance(image_path):
    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.moveaxis(rgb, 0, -1)
        rgb = rgb.astype(np.float32) / 255.0

    tile_basename = os.path.splitext(os.path.basename(image_path))[0]
    parts = tile_basename.split("_")
    row, col = parts[-2], parts[-1]

    search_pattern = os.path.join(
        "/Volumes/toshiba/UNET/tiles/distance/",
        f"*_{row}_{col}_distance.tif"
    )
    matches = glob.glob(search_pattern)

    if not matches:
        raise FileNotFoundError(f"Distance raster not found for tile: {tile_basename}")

    dist_path = matches[0]

    with rasterio.open(dist_path) as dist_src:
        dist = dist_src.read(1)
        dist = np.expand_dims(dist, axis=-1)
        dist = dist.astype(np.float32)
        dist = dist / dist.max() if dist.max() > 0 else dist

    combined = np.concatenate([rgb, dist], axis=-1)
    return combined

# === CUSTOM WEIGHTED LOSS FUNCTION ===
# Assign higher importance to classes 0 and 1
class_weights = tf.constant([3.0, 3.0, 1.0, 0.5])  # [open water, landfast ice, drift ice, cloud/unknown]

def weighted_categorical_crossentropy(y_true, y_pred):
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss * weights
