import os
import re
import glob
import numpy as np
import tensorflow as tf
import rasterio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# === CONFIG ===
model_path = r"C:\Users\as1612\Desktop\unet-4class-RGBD.keras"
tile_dir = r"D:\UNET\tiles_infer\images"
distance_dir = r"D:\UNET\tiles_infer\distance"
tile_size = 320
stride = 240  # must match tiling.py
pred_tile_dir = r"D:\UNET\tiles\predicted_tiles"
stitched_output_dir = r"D:\UNET\predictions"
target_scene = "Sentinel2_Wainwright_AK_20220418"
max_canvas_dim = 20000

os.makedirs(pred_tile_dir, exist_ok=True)
os.makedirs(stitched_output_dir, exist_ok=True)

color_map = {
    0: (0, 0, 255),       # Open Water – Blue
    1: (255, 255, 255),   # Landfast Ice – White
    2: (0, 255, 255),     # Drift Ice – Cyan
    3: (255, 0, 255),     # Transition Ice – Magenta
}

# === LOAD MODEL ===
print("📦 Loading model...")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded!")

# === PREDICT ===
tiles = sorted(glob.glob(os.path.join(tile_dir, f"{target_scene}_*.tif")))
print(f"🔍 Found {len(tiles)} tiles for {target_scene}")

for tile_path in tqdm(tiles, desc=f"🧩 Predicting tiles"):
    base = os.path.basename(tile_path).replace('.tif', '')
    dist_path = os.path.join(distance_dir, f"{base}_distance.tif")
    if not os.path.exists(dist_path):
        print(f"⚠️ Missing distance raster for {base}, skipping.")
        continue

    try:
        with rasterio.open(tile_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
            rgb_uint8 = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        with rasterio.open(dist_path) as dsrc:
            dist = dsrc.read(1).astype(np.float32)
            dist = np.expand_dims(dist, axis=0)
            dist /= np.max(dist) if np.max(dist) != 0 else 1

        img = np.concatenate([rgb, dist], axis=0)
        img = np.moveaxis(img, 0, -1)

        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_mask = np.argmax(pred.squeeze(), axis=-1).astype(np.uint8)

        Image.fromarray(pred_mask).save(os.path.join(pred_tile_dir, f"{base}_pred.png"))
        Image.fromarray(rgb_uint8).save(os.path.join(pred_tile_dir, f"{base}_rgb.png"))

    except Exception as e:
        print(f"❌ Failed to process {base}: {e}")

# === STITCHING WITH SOFT BLENDING ===
print("\n🧵 Grouping tiles for stitching...")
scene_tiles = []
for pred_tile in sorted(glob.glob(os.path.join(pred_tile_dir, f"{target_scene}_*_pred.png"))):
    base = os.path.basename(pred_tile).replace("_pred.png", "")
    match = re.match(rf"{target_scene}_(\d+)_(\d+)", base)
    if match:
        row, col = map(int, match.groups())
        scene_tiles.append((row, col, base))

if not scene_tiles:
    print(f"⚠️ No tiles to stitch for {target_scene}")
else:
    print(f"\n🧵 Stitching scene {target_scene} with {len(scene_tiles)} tiles")
    rows = [r for r, c, _ in scene_tiles]
    cols = [c for r, c, _ in scene_tiles]
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    height = (max_row - min_row + 1) * stride + (tile_size - stride)
    width = (max_col - min_col + 1) * stride + (tile_size - stride)

    print(f"🧠 Canvas size: {height}px x {width}px")
    if height > max_canvas_dim or width > max_canvas_dim:
        print("🚫 Aborting stitching — canvas too large.")
    else:
        pred_sum = np.zeros((height, width, 4), dtype=np.float32)
        weight_sum = np.zeros((height, width, 1), dtype=np.float32)
        rgb_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for row, col, base in scene_tiles:
            pred_path = os.path.join(pred_tile_dir, f"{base}_pred.png")
            rgb_path = os.path.join(pred_tile_dir, f"{base}_rgb.png")
            pred = np.array(Image.open(pred_path))
            rgb = np.array(Image.open(rgb_path))

            y0 = (row - min_row) * stride
            x0 = (col - min_col) * stride

            # One-hot encode prediction
            one_hot = np.eye(4)[pred]  # shape: (tile_size, tile_size, 4)
            weight = np.ones((tile_size, tile_size, 1), dtype=np.float32)

            pred_sum[y0:y0+tile_size, x0:x0+tile_size] += one_hot * weight
            weight_sum[y0:y0+tile_size, x0:x0+tile_size] += weight
            rgb_canvas[y0:y0+tile_size, x0:x0+tile_size] = rgb

        blended_probs = pred_sum / np.clip(weight_sum, 1e-6, None)
        stitched_pred = np.argmax(blended_probs, axis=-1).astype(np.uint8)

        # === Overlay
        overlay = np.zeros_like(rgb_canvas)
        for cls, color in color_map.items():
            overlay[stitched_pred == cls] = color
        overlayed = (0.6 * rgb_canvas + 0.4 * overlay).astype(np.uint8)
        side_by_side = np.concatenate([rgb_canvas, overlayed], axis=1)

        # === Add Legend
        legend_height = 60
        legend = Image.new("RGB", (side_by_side.shape[1], legend_height), (0, 0, 0))
        draw = ImageDraw.Draw(legend)
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        labels = [("Open Water", 0), ("Landfast Ice", 1), ("Drift Ice", 2), ("Transition Ice", 3)]
        spacing = 160
        for i, (label, cls) in enumerate(labels):
            x = 10 + i * spacing
            y = (legend_height - 20) // 2
            draw.rectangle([x, y, x + 20, y + 20], fill=color_map[cls])
            draw.text((x + 28, y), label, fill=(255, 255, 255), font=font)

        final_img = Image.new("RGB", (side_by_side.shape[1], side_by_side.shape[0] + legend_height))
        final_img.paste(Image.fromarray(side_by_side), (0, 0))
        final_img.paste(legend, (0, side_by_side.shape[0]))

        out_img_path = os.path.join(stitched_output_dir, f"{target_scene}_stitched_overlay.png")
        final_img.save(out_img_path)
        print(f"🖼️ Saved: {out_img_path}")

        # === Save stitched mask as GeoTIFF
        out_mask_path = os.path.join(stitched_output_dir, f"{target_scene}_stitched_prediction.tif")
        with rasterio.open(tiles[0]) as ref:
            transform = ref.transform
            crs = ref.crs
        with rasterio.open(
            out_mask_path, 'w',
            driver='GTiff',
            height=stitched_pred.shape[0],
            width=stitched_pred.shape[1],
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(stitched_pred, 1)

        print(f"🗺️ Saved: {out_mask_path}")
