import os
import glob
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

# === CONFIG ===
pred_tile_dir = r"D:\UNET\tiles\predicted_tiles"
stitched_dir = r"D:\UNET\stitched_predictions"
tile_size = 320

os.makedirs(stitched_dir, exist_ok=True)

# === Identify Scenes ===
def get_scene_names():
    tiles = glob.glob(os.path.join(pred_tile_dir, "*_pred.png"))
    scenes = set("_".join(os.path.basename(t).split("_")[:-3]) for t in tiles)
    return sorted(list(scenes))

scenes = get_scene_names()
print(f"🔍 Found {len(scenes)} scenes: {scenes}")

# === Stitch Each Scene ===
for scene in scenes:
    print(f"\n🧵 Stitching scene: {scene}")

    # Get all _pred tiles
    tile_paths = glob.glob(os.path.join(pred_tile_dir, f"{scene}_*_pred.png"))
    if not tile_paths:
        print(f"⚠️ No tiles found for {scene}, skipping.")
        continue

    row_indices, col_indices = [], []
    tile_dict = {}

    # Parse row/col offsets
    for path in tile_paths:
        base = os.path.basename(path).replace("_pred.png", "")
        parts = base.split("_")
        row, col = int(parts[-2]), int(parts[-1])
        tile = np.array(Image.open(path))
        tile_dict[(row, col)] = tile
        row_indices.append(row)
        col_indices.append(col)

    # Determine canvas size
    max_row = max(row_indices)
    max_col = max(col_indices)
    height = max_row + tile_size
    width = max_col + tile_size
    stitched = np.full((height, width), 255, dtype=np.uint8)  # 255 = nodata

    # Place tiles
    for (row, col), tile in tqdm(tile_dict.items(), desc=f"Stitching {scene}"):
        stitched[row:row + tile_size, col:col + tile_size] = tile

    # Save stitched raster
    stitched_path = os.path.join(stitched_dir, f"{scene}_stitched_pred.tif")
    with rasterio.open(
        stitched_path,
        "w",
        driver="GTiff",
        height=stitched.shape[0],
        width=stitched.shape[1],
        count=1,
        dtype=np.uint8,
        transform=from_origin(0, 0, 1, 1)
    ) as dst:
        dst.write(stitched, 1)

    print(f"✅ Stitched prediction saved: {stitched_path}")
