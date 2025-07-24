import os
import numpy as np
import rasterio
from PIL import Image

# === CONFIG ===
stitched_dir = r"D:\UNET\stitched_predictions"
rgb_base_dir = r"D:\W"  # Base folder where RGB scenes are stored by year
output_dir = r"D:\UNET\stitched_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Color map for classes
color_map = {
    0: (0, 0, 255),       # Open Water - Blue
    1: (255, 255, 255),   # Landfast Ice - White
    2: (0, 255, 255),     # Drift Ice - Cyan
    3: (255, 0, 255),     # Transition Ice - Magenta
}

def apply_colormap(mask):
    """Convert class mask to RGB color visualization."""
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colorized[mask == class_id] = color
    return colorized

def find_rgb_path(scene_name):
    """Guess the RGB path based on scene name and year subfolder."""
    year = scene_name.split("_")[-1][:4]
    rgb_path = os.path.join(rgb_base_dir, year, f"{scene_name}.tif")
    return rgb_path if os.path.exists(rgb_path) else None

# === LOOP OVER STITCHED PREDICTIONS ===
for filename in os.listdir(stitched_dir):
    if not filename.endswith("_stitched_pred.tif"):
        continue
    
    scene_name = filename.replace("_stitched_pred.tif", "")
    stitched_path = os.path.join(stitched_dir, filename)
    colorized_path = os.path.join(output_dir, f"{scene_name}_colorized.png")
    overlay_path = os.path.join(output_dir, f"{scene_name}_overlay.png")
    
    print(f"🎨 Processing {scene_name}...")

    # Load stitched mask
    with rasterio.open(stitched_path) as src:
        mask = src.read(1).astype(np.uint8)

    # Colorized mask
    colorized = apply_colormap(mask)
    Image.fromarray(colorized).save(colorized_path)
    print(f"✅ Saved colorized mask: {colorized_path}")

    # Try to find original RGB
    rgb_path = find_rgb_path(scene_name)
    if rgb_path and os.path.exists(rgb_path):
        with rasterio.open(rgb_path) as src:
            rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        
        # Resize prediction if needed
        if rgb.shape[:2] != colorized.shape[:2]:
            colorized_resized = np.array(Image.fromarray(colorized).resize((rgb.shape[1], rgb.shape[0])))
        else:
            colorized_resized = colorized

        # Blend (50/50)
        overlay = Image.blend(Image.fromarray(rgb), Image.fromarray(colorized_resized), alpha=0.5)
        overlay.save(overlay_path)
        print(f"✅ Saved overlay: {overlay_path}")
    else:
        print(f"⚠️ RGB scene not found for {scene_name}. Skipping overlay.")
