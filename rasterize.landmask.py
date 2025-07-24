import os
import rasterio
from rasterio import features
from shapely.geometry import mapping, box
import geopandas as gpd
import numpy as np

# === CONFIG ===
landmass_path = r"D:\UNET\cb_2022_us_state_5m\cb_2022_us_state_5m.shp"

image_paths = [
    r"D:\W\2019\Sentinel2_Wainwright_AK_20190301.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220310.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220418.tif",
    r"D:\W\2023\Sentinel2_Wainwright_AK_20230423.tif",
    r"D:\W\2024\Sentinel2_Wainwright_AK_20240306.tif"
]
output_dir = r"D:\UNET\landmask"
os.makedirs(output_dir, exist_ok=True)

land_value = 1
background_value = 0
dtype = "uint8"

# === MAIN FUNCTION ===
def rasterize_landmask(rgb_path, land_gdf, output_path):
    with rasterio.open(rgb_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        shape = (src.height, src.width)

    # Reproject land polygons to image CRS
    land_proj = land_gdf.to_crs(crs)

    # Clip polygon to image bounds (expanded slightly to ensure overlap)
    buffer_deg = 0.1  # roughly 10 km depending on lat
    scene_box = box(bounds.left - buffer_deg, bounds.bottom - buffer_deg,
                    bounds.right + buffer_deg, bounds.top + buffer_deg)
    clipped = land_proj.clip(scene_box)

    if clipped.empty:
        print(f"⚠️ No intersection between land polygons and {os.path.basename(rgb_path)} — skipping.")
        return

    # Rasterize
    shapes = [(mapping(geom), land_value) for geom in clipped.geometry if geom.is_valid]
    mask = features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=background_value,
        dtype=dtype
    )

    # Save
    meta.update({"count": 1, "dtype": dtype})
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask, 1)

    print(f"✅ Saved: {output_path}")

# === DRIVER ===
if __name__ == "__main__":
    land_gdf = gpd.read_file(landmass_path)

    for rgb_path in image_paths:
        fname = os.path.basename(rgb_path)
        out_path = os.path.join(output_dir, fname.replace(".tif", "_landmask.tif"))

        try:
            rasterize_landmask(rgb_path, land_gdf, out_path)
        except Exception as e:
            print(f"❌ Error for {fname}: {e}")
