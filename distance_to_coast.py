import os
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping

# === INPUT PATHS ===
image_paths = [
    r"D:\W\2019\Sentinel2_Wainwright_AK_20190301.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220310.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220418.tif",
    r"D:\W\2023\Sentinel2_Wainwright_AK_20230423.tif",
    r"D:\W\2024\Sentinel2_Wainwright_AK_20240306.tif"
]

# Use simplified land polygon (not the complex Alaska coastline)
coastline_path = r"D:\UNET\cb_2022_us_state_5m\cb_2022_us_state_5m.shp"
output_dir = r"D:\UNET\distance_rasters"
os.makedirs(output_dir, exist_ok=True)

# === LOAD COASTLINE SHAPEFILE ===
coast_gdf = gpd.read_file(coastline_path)
if coast_gdf.empty:
    raise ValueError("🚫 Coastline shapefile is empty.")
print(f"🌊 Loaded coastline layer with {len(coast_gdf)} features.")

# === LOOP OVER EACH IMAGE ===
for img_path in image_paths:
    base = os.path.basename(img_path).replace(".tif", "")
    print(f"\n🔧 Processing: {base}")

    with rasterio.open(img_path) as src:
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

        # Reproject coastline to match image CRS
        coast_proj = coast_gdf.to_crs(crs)

        # Validate geometries
        coast_proj = coast_proj[coast_proj.geometry.is_valid & ~coast_proj.geometry.is_empty]
        if coast_proj.empty:
            print(f"⚠️ No valid geometries after reprojection for {base}. Skipping.")
            continue

        # Rasterize coastline mask
        coast_shapes = ((geom, 1) for geom in coast_proj.geometry)
        coast_mask = rasterio.features.rasterize(
            coast_shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Compute Euclidean distance transform from coastline (in pixels)
        dist = distance_transform_edt(1 - coast_mask).astype(np.float32)

        # Save the distance raster
        out_path = os.path.join(output_dir, f"{base}_distance_to_coast.tif")
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=shape[0],
            width=shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(dist, 1)

        print(f"✅ Saved: {out_path}")
