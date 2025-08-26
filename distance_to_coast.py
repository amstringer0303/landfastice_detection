import os
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import distance_transform_edt

# === INPUT IMAGE PATHS ===
image_paths = [
    r"D:\W\2019\Sentinel2_Wainwright_AK_20190301.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220310.tif",
    r"D:\W\2022\Sentinel2_Wainwright_AK_20220418.tif",
    r"D:\W\2023\Sentinel2_Wainwright_AK_20230423.tif",
    r"D:\W\2024\Sentinel2_Wainwright_AK_20240306.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240303.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240415.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240326.tif",
    r"D:\A\drive-download-20250501T155129Z-001\Sentinel2_Arviat_NU_20200312.tif",
    r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240427.tif",
    r"D:\barrow\Sentinel2_Barrow_AK_20220401.tif",
    r"D:\barrow\Sentinel2_Barrow_AK_20220416(1).tif"
]

# === COASTLINE SHAPEFILES ===
us_landmass_path = r"D:\UNET\cb_2022_us_state_5m\cb_2022_us_state_5m.shp"
canada_landmass_path = r"D:\GSHHS_shp\f\GSHHS_f_L1.shp"

# === OUTPUT DIRECTORY ===
output_dir = r"D:\UNET\distance_rasters"
os.makedirs(output_dir, exist_ok=True)

# === SELECT LANDMASS BASED ON IMAGE NAME ===
def get_landmass_path(img_path):
    if "Arviat" in img_path or "NU" in img_path:
        return canada_landmass_path
    else:
        return us_landmass_path

# === PROCESSING LOOP ===
for img_path in image_paths:
    base = os.path.basename(img_path).replace(".tif", "")
    print(f"\n📏 Processing distance raster for: {base}")

    # Determine appropriate landmass shapefile
    landmass_path = get_landmass_path(img_path)
    coast_gdf = gpd.read_file(landmass_path)
    if coast_gdf.empty:
        print(f"🚫 Skipping {base} — shapefile at {landmass_path} is empty.")
        continue

    with rasterio.open(img_path) as src:
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    # Reproject coastline to match image CRS
    coast_proj = coast_gdf.to_crs(crs)
    coast_proj = coast_proj[coast_proj.geometry.is_valid & ~coast_proj.geometry.is_empty]
    if coast_proj.empty:
        print(f"⚠️ No valid coastline geometries after reprojection for {base}. Skipping.")
        continue

    # Rasterize coastlines
    shapes = ((geom, 1) for geom in coast_proj.geometry)
    coast_mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # Compute distance transform (in pixels)
    distance = distance_transform_edt(1 - coast_mask).astype(np.float32)

    # Save distance raster
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
        dst.write(distance, 1)

    print(f"✅ Saved: {out_path}")
