import os
import glob
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape, MultiPolygon, Polygon, LineString, mapping
import geopandas as gpd
from tqdm import tqdm

# === CONFIGURATION ===
stitched_dir = r"D:\UNET\stitched_predictions"  # Folder with stitched *_pred.tif
output_dir = r"D:\UNET\ice_edges"  # Folder to save GeoJSON ice edge lines
os.makedirs(output_dir, exist_ok=True)

# Class ID for landfast ice
LANDF_AST_CLASS = 1

# === FUNCTION TO EXTRACT OUTER EDGE ===
def get_outer_boundary(mask, transform):
    """
    Extracts the outer boundary (seaward edge) of the largest landfast ice polygon.
    """
    # Convert landfast mask (1) to shapes
    shapes_gen = features.shapes(mask.astype(np.uint8), mask=(mask == LANDF_AST_CLASS), transform=transform)
    polygons = [shape(geom) for geom, value in shapes_gen if value == LANDF_AST_CLASS and shape(geom).area > 0]

    if not polygons:
        return None

    # Merge all polygons into one MultiPolygon
    merged = MultiPolygon(polygons)
    merged = merged.buffer(0)  # Fix invalid geometry, if any

    # Get the exterior boundary line (seaward edge)
    if isinstance(merged, Polygon):
        boundary = merged.exterior
    else:
        # MultiPolygon: choose the largest polygon (by area)
        largest_poly = max(merged.geoms, key=lambda p: p.area)
        boundary = largest_poly.exterior

    return boundary

# === MAIN PROCESS ===
stitched_files = sorted(glob.glob(os.path.join(stitched_dir, "*_stitched_pred.tif")))

print(f"🔍 Found {len(stitched_files)} stitched prediction files.")

for tif_path in tqdm(stitched_files, desc="Extracting Ice Edges"):
    scene_name = os.path.basename(tif_path).replace("_stitched_pred.tif", "")
    out_geojson = os.path.join(output_dir, f"{scene_name}_ice_edge.geojson")

    with rasterio.open(tif_path) as src:
        pred = src.read(1)  # Predicted classes
        transform = src.transform
        crs = src.crs

    # Extract the outer boundary
    boundary = get_outer_boundary(pred, transform)

    if boundary is None:
        print(f"⚠️ No landfast ice (class 1) found in {scene_name}, skipping.")
        continue

    # Save as GeoJSON
    gdf = gpd.GeoDataFrame({"scene": [scene_name]}, geometry=[LineString(boundary)], crs=crs)
    gdf.to_file(out_geojson, driver="GeoJSON")
    print(f"✅ Saved ice edge: {out_geojson}")
