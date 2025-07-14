import os
import rasterio
import rasterio.features
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping

# === INPUT CONFIGURATION ===
image_paths = [
    "/Volumes/toshiba/W/2019/Sentinel2_Wainwright_AK_20190301.tif",
    "/Volumes/toshiba/W/2022/Sentinel2_Wainwright_AK_20220310.tif",
    "/Volumes/toshiba/W/2022/Sentinel2_Wainwright_AK_20220418_fixed.tif",
    "/Volumes/toshiba/W/2023/Sentinel2_Wainwright_AK_20230423.tif",
    "/Volumes/toshiba/W/2024/Sentinel2_Wainwright_AK_20240306.tif"
]

label_paths = {
    "20190301": "/Volumes/toshiba/UNET/20190301.geojson",
    "20220310": "/Volumes/toshiba/UNET/20220310.geojson",
    "20220418": "/Volumes/toshiba/UNET/20220418.geojson",
    "20230423": "/Volumes/toshiba/UNET/20230423.geojson",
    "20240306": "/Volumes/toshiba/UNET/20240306.geojson"
}

coastline_path = "/Volumes/toshiba/UNET/ALASKA_63360_LN.geojson"
output_dir = "/Volumes/toshiba/UNET/masks_rasterized/"
NODATA_VALUE = 255
os.makedirs(output_dir, exist_ok=True)

# === LOAD AND PREPROCESS COASTLINE ===
coast_gdf = gpd.read_file(coastline_path)
coastline_buffered = coast_gdf.buffer(-10)
print("🌊 Coastline loaded and buffered.")

# === RASTERIZE LOOP ===
for image_path in image_paths:
    base = os.path.basename(image_path)
    img_name = base.replace("Sentinel2_Wainwright_AK_", "").replace("_fixed.tif", "").replace(".tif", "")
    label_path = label_paths.get(img_name)

    print(f"\n🔄 Rasterizing: {img_name}")

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue
    if not os.path.exists(label_path):
        print(f"❌ Label not found: {label_path}")
        continue

    with rasterio.open(image_path) as src:
        image_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs
        print(f"📐 Image dimensions: {image_shape}, CRS: {crs}")

        if transform == rasterio.Affine.identity:
            print(f"⚠️ {img_name} has identity transform — likely missing spatial placement.")
        if crs is None:
            print(f"⚠️ {img_name} has no CRS — image is not georeferenced.")

    gdf = gpd.read_file(label_path).to_crs(crs)
    coast_buffered_proj = gpd.GeoDataFrame(geometry=coastline_buffered, crs=coast_gdf.crs).to_crs(crs)

    if 'class_id' not in gdf.columns:
        raise ValueError(f"GeoJSON for {img_name} must have a 'class_id' column.")

    valid_gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
    if valid_gdf.empty:
        print(f"⚠️ No valid geometries in {label_path} — skipping.")
        continue

    expected_classes = set(range(4))  # 0 to 3
    found_classes = set(valid_gdf["class_id"].unique())
    if not found_classes.issubset(expected_classes):
        raise ValueError(f"❌ Unexpected class IDs in {img_name}: {found_classes - expected_classes}")

    print(f"✅ {len(valid_gdf)} valid polygons retained for rasterization.")

    # Rasterize class polygons
    class_shapes = ((geom, int(class_id)) for geom, class_id in zip(valid_gdf.geometry, valid_gdf["class_id"]))
    class_mask = rasterio.features.rasterize(
        shapes=class_shapes,
        out_shape=image_shape,
        transform=transform,
        fill=NODATA_VALUE,
        dtype=np.uint8
    )

    # Apply coastline exclusion mask
    coast_shapes = ((geom, NODATA_VALUE) for geom in coast_buffered_proj.geometry if geom.is_valid)
    coast_mask = rasterio.features.rasterize(
        shapes=coast_shapes,
        out_shape=image_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )
    class_mask[coast_mask == NODATA_VALUE] = NODATA_VALUE

    # Pixel value summary
    unique, counts = np.unique(class_mask, return_counts=True)
    print("📊 Pixel distribution:", dict(zip(unique, counts)))

    # Save to GeoTIFF
    out_path = os.path.join(output_dir, f"{img_name}_mask.tif")
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=class_mask.shape[0],
        width=class_mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        nodata=NODATA_VALUE
    ) as dst:
        dst.write(class_mask, 1)

    print(f"✅ Saved: {out_path}")
