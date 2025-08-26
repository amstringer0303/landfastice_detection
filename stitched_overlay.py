import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

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

stitched_dir = r"D:\UNET\stitched_predictions"
output_dir = r"D:\UNET\overlays"
os.makedirs(output_dir, exist_ok=True)

for scene, paths in scenes.items():
    pred_path = os.path.join(stitched_dir, f"{scene}_stitched_pred.tif")
    if not os.path.exists(pred_path):
        print(f"❌ Prediction not found for {scene}, skipping.")
        continue

    with rasterio.open(pred_path) as src:
        pred = src.read(1)
    
    with rasterio.open(paths["rgb"]) as src:
        rgb = src.read([1, 2, 3])
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())  # normalize

    plt.figure(figsize=(12, 12))
    plt.imshow(rgb)
    plt.imshow(pred, cmap='jet', alpha=0.5)
    plt.title(scene, fontsize=16)
    plt.axis('off')

    out_path = os.path.join(output_dir, f"{scene}_overlay.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Overlay saved: {out_path}")
