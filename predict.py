import os
import glob
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

# === CONFIG ===
model_path = r"D:\UNET\models\unet_landfastice_distance.keras"
tile_dir = r"D:\UNET\tiles\images"
distance_dir = r"D:\UNET\tiles\distance"
pred_tile_dir = r"D:\UNET\tiles\predicted_tiles"
tile_size = 320

# Ensure output directory exists
os.makedirs(pred_tile_dir, exist_ok=True)

# === Custom Weighted Loss (match training) ===
weights = tf.constant([2.5, 5.0, 1.0, 0.5], dtype=tf.float32)

def weighted_cce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weights_map = tf.reduce_sum(y_true * weights, axis=-1)
    return loss * weights_map

# === Load Trained Model ===
print("📦 Loading model...")
model = tf.keras.models.load_model(model_path, custom_objects={"weighted_cce": weighted_cce})
print("✅ Model loaded.")

# === Colormap for visualization ===
color_map = {
    0: (0, 0, 255),       # Open Water – Blue
    1: (255, 255, 255),   # Landfast Ice – White
    2: (0, 255, 255),     # Drift Ice – Cyan
    3: (255, 0, 255),     # Transition Ice – Magenta
}

def apply_colormap(mask):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb[mask == class_id] = color
    return rgb

# === Predict on Each Tile ===
tile_paths = sorted(glob.glob(os.path.join(tile_dir, "*.tif")))
print(f"\n🔍 Found {len(tile_paths)} tiles.")

for tile_path in tqdm(tile_paths, desc="🧩 Predicting tiles"):
    base = os.path.basename(tile_path).replace(".tif", "")
    dist_path = os.path.join(distance_dir, f"{base}_distance.tif")

    if not os.path.exists(dist_path):
        print(f"⚠️ Missing distance raster for {base}, skipping.")
        continue

    try:
        # Load RGB
        with rasterio.open(tile_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
            rgb_vis = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)

        # Load distance
        with rasterio.open(dist_path) as dsrc:
            dist = dsrc.read(1).astype(np.float32)
            dist = np.expand_dims(dist, axis=0)
            dist /= np.max(dist) if np.max(dist) > 0 else 1.0

        # Combine channels
        image = np.concatenate([rgb, dist], axis=0)
        image = np.transpose(image, (1, 2, 0))  # CHW → HWC

        # Predict
        pred = model.predict(np.expand_dims(image, axis=0), verbose=0)
        pred_mask = np.argmax(pred.squeeze(), axis=-1).astype(np.uint8)

        # Save prediction mask (raw class ids)
        pred_mask_path = os.path.join(pred_tile_dir, f"{base}_pred.png")
        Image.fromarray(pred_mask).save(pred_mask_path)

        # Save RGB image (for reference)
        rgb_out_path = os.path.join(pred_tile_dir, f"{base}_rgb.png")
        Image.fromarray(rgb_vis).save(rgb_out_path)

        # Save colorized mask
        color_mask = apply_colormap(pred_mask)
        color_path = os.path.join(pred_tile_dir, f"{base}_overlay.png")
        Image.fromarray(color_mask).save(color_path)

    except Exception as e:
        print(f"❌ Failed to process {base}: {e}")
