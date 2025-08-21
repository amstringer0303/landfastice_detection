import os
import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np

# === CONFIGURATION ===
scenes = {
    # WAINWRIGHT
    "Sentinel2_Wainwright_AK_20190301": {
        "rgb": r"D:\W\2019\Sentinel2_Wainwright_AK_20190301.tif",
        "geojson": r"D:\UNET\20190301.geojson",
        "landmask": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20190301_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20220310": {
        "rgb": r"D:\W\2022\Sentinel2_Wainwright_AK_20220310.tif",
        "geojson": r"D:\UNET\20220310.geojson",
        "landmask": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20220310_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20220418": {
        "rgb": r"D:\W\2022\Sentinel2_Wainwright_AK_20220418.tif",
        "geojson": r"D:\UNET\20220418.geojson",
        "landmask": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20220418_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20230423": {
        "rgb": r"D:\W\2023\Sentinel2_Wainwright_AK_20230423.tif",
        "geojson": r"D:\UNET\20230423.geojson",
        "landmask": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20230423_landmask.tif"
    },
    "Sentinel2_Wainwright_AK_20240306": {
        "rgb": r"D:\W\2024\Sentinel2_Wainwright_AK_20240306.tif",
        "geojson": r"D:\UNET\20240306.geojson",
        "landmask": r"D:\UNET\landmask\Sentinel2_Wainwright_AK_20240306_landmask.tif"
    },

    # ARVIAT
    "Sentinel2_Arviat_NU_20240303": {
        "rgb": r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240303.tif",
        "geojson": r"D:\A\Arviat_NU_0303.gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Arviat_NU_20240303_landmask.tif"
    },
    "Sentinel2_Arviat_NU_20240415": {
        "rgb": r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240415.tif",
        "geojson": r"D:\A\Arviat_NU_20240415.gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Arviat_NU_20240415_landmask.tif"
    },
    "Sentinel2_Arviat_NU_20240326": {
        "rgb": r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240326.tif",
        "geojson": r"D:\A\Arviat_NU_20240326.gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Arviat_NU_20240326_landmask.tif"
    },
    "Sentinel2_Arviat_NU_20200312": {
        "rgb": r"D:\A\drive-download-20250501T155129Z-001\Sentinel2_Arviat_NU_20200312.tif",
        "geojson": r"D:\A\Arviat_NU_20200312.gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Arviat_NU_20200312_landmask.tif"
    },
    "Sentinel2_Arviat_NU_20240427": {
        "rgb": r"D:\A\drive-download-20250811T063631Z-1-001\Sentinel2_Arviat_NU_20240427.tif",
        "geojson": r"D:\A\Arviat_NU_0427.gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Arviat_NU_20240427_landmask.tif"
    },

    # BARROW
    "Sentinel2_Barrow_AK_20220401": {
        "rgb": r"D:\barrow\Sentinel2_Barrow_AK_20220401.tif",
        "geojson": r"D:\barrow\Barrow_20220401.gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Barrow_AK_20220401_landmask.tif"
    },
    "Sentinel2_Barrow_AK_20220416": {
        "rgb": r"D:\barrow\Sentinel2_Barrow_AK_20220416(1).tif",
        "geojson": r"D:\barrow\Barrow_20220416(1).gpkg",
        "landmask": r"D:\UNET\landmask\Sentinel2_Barrow_AK_20220416_landmask.tif"
    }
}

output_dir = r"D:\UNET\masks_rasterized"
os.makedirs(output_dir, exist_ok=True)

# === LOOP THROUGH EACH SCENE ===
for scene, paths in scenes.items():
    print(f"\n🖍️ Rasterizing: {scene}")

    with rasterio.open(paths["rgb"]) as ref:
        shape = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs

    # Load GeoJSON or GPKG and ensure CRS matches RGB
    gdf = gpd.read_file(paths["geojson"]).to_crs(crs)

    # Initialize label_mask with 255 (NODATA)
    label_mask = np.full(shape, 255, dtype=np.uint8)

    # Rasterize each class (e.g., 0 = open water, 1 = landfast ice)
    for class_id in sorted(gdf["class_id"].unique()):
        class_polys = gdf[gdf["class_id"] == class_id]
        if not class_polys.empty:
            shapes = ((geom, class_id) for geom in class_polys.geometry if geom.is_valid)
            mask_layer = rasterio.features.rasterize(
                shapes=shapes,
                out_shape=shape,
                transform=transform,
                fill=255,
                dtype=np.uint8
            )
            label_mask[mask_layer == class_id] = class_id

    # Apply land mask
    with rasterio.open(paths["landmask"]) as land_src:
        if land_src.crs != crs:
            raise ValueError(f"CRS mismatch: {scene} — landmask {land_src.crs}, RGB {crs}")
        land_mask = land_src.read(1)
        label_mask[land_mask == 1] = 255  # Set land to NODATA

    # Save output mask
    out_path = os.path.join(output_dir, f"{scene.split('_')[-1]}_mask.tif")
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(label_mask, 1)

    # Pixel count summary
    unique, counts = np.unique(label_mask, return_counts=True)
    print("📊 Final pixel distribution:", dict(zip(unique, counts)))
    print(f"✅ Saved: {out_path}")
