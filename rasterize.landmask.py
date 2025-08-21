import os
import rasterio
from rasterio import features
from shapely.geometry import mapping, box
import geopandas as gpd
import numpy as np

# === INPUT: RGB IMAGE PATHS ===
image_paths = [
    # Wainwright
    r"D:\W\2019\Sentinel2_Wainwright_AK_20190301.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220310.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220418.tif",
    r"D:\W\2023\Sentinel2_Wainwright_AK_20230423.tif",
    r"D:\W\2024\Sentinel2_Wainwright_AK_20240306.tif",

    # Arviat
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240303.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240415.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240326.tif",
    r"D:\A\drive-download-20250501T155129Z-001\Sentinel2_Arviat_NU_20200312.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240427.tif",

    # Barrow
    r"D:\barrow\Sentinel2_Barrow_AK_20220401.tif",
    r"D:\barrow\Sentinel2_Barrow_AK_20220416(1).tif"
]

# === COASTLINE SHAPEFILES ===
us_landmass_path = r"D:\UNET\cb_2022_us_state_5m\cb_2022_us_state_5m.shp"
canada_landmass_path = r"D:\GSHHS_shp\f\GSHHS_f_L1.shp"

# === OUTPUT DIRECTORY ===
output_dir = r"D:\UNET\landmask"
os.makedirs(output_dir, exist_ok=True)

# === SETTINGS ===
land_value = 1
background_value = 0
dtype = "uint8"

# === SELECT LANDMASS BASED ON IMAGE LOCATION ===
def get_landmass_path(rgb_path):
    if "Arviat" in rgb_path or "NU" in rgb_path:
        return canada_landmass_path
    else:
        return us_landmass_path

# === RASTERIZE FUNCTION ===
def rasterize_landmask(rgb_path, land_gdf, output_path):
    with rasterio.open(rgb_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        shape = (src.height, src.width)

    # Reproject landmass to match image CRS
    land_proj = land_gdf.to_crs(crs)

    # Clip landmass to image bounds (with buffer)
    buffer_deg = 0.1
    scene_box = box(bounds.left - buffer_deg, bounds.bottom - buffer_deg,
                    bounds.right + buffer_deg, bounds.top + buffer_deg)
    clipped = land_proj.clip(scene_box)

    if clipped.empty:
        print(f"⚠️ No intersection between land polygons and {os.path.basename(rgb_path)} — skipping.")
        return

    # Rasterize clipped land
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

# === MAIN DRIVER ===
if __name__ == "__main__":
    for rgb_path in image_paths:
        landmass_path = get_landmass_path(rgb_path)
        land_gdf = gpd.read_file(landmass_path)

        fname = os.path.basename(rgb_path).replace("(1)", "")  # optional cleanup
        out_path = os.path.join(output_dir, fname.replace(".tif", "_landmask.tif"))

        try:
            rasterize_landmask(rgb_path, land_gdf, out_path)
        except Exception as e:
            print(f"❌ Error for {fname}: {e}")
