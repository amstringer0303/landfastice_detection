import os
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import distance_transform_edt
from shapely.geometry import mapping

# === INPUT PATHS ===
image_paths = [
    "/Volumes/toshiba/W/2019/Sentinel2_Wainwright_AK_20190301.tif",
    "/Volumes/toshiba/W/2022/Sentinel2_Wainwright_AK_20220310.tif",
    "/Volumes/toshiba/W/2022/Sentinel2_Wainwright_AK_20220418_fixed.tif",
    "/Volumes/toshiba/W/2023/Sentinel2_Wainwright_AK_20230423.tif",
    "/Volumes/toshiba/W/2024/Sentinel2_Wainwright_AK_20240306.tif"
]
coastline_path = "/Volumes/toshiba/UNET/ALASKA_63360_LN.geojson"
output_dir = os.path.expanduser("~/Desktop/distance_rasters/")
os.makedirs(output_dir, exist_ok=True)

# === LOAD COASTLINE SHAPEFILE ===
coast_gdf = gpd.read_file(coastline_path)
print("🌊 Loaded coastline polygons.")

# === LOOP OVER EACH IMAGE ===
for img_path in image_paths:
    base = os.path.basename(img_path).replace(".tif", "")
    print(f"\n🔧 Processing: {base}")

    with rasterio.open(img_path) as src:
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

        # Reproject coastline to match image
        coast_proj = coast_gdf.to_crs(crs)
        coast_shapes = ((geom, 1) for geom in coast_proj.geometry if geom.is_valid)

        # Rasterize coastline binary mask
        coast_mask = rasterio.features.rasterize(
            coast_shapes, out_shape=shape, transform=transform, fill=0, dtype=np.uint8
        )

        # Invert and compute Euclidean distance transform
        dist = distance_transform_edt(1 - coast_mask)
        dist = dist.astype(np.float32)

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
