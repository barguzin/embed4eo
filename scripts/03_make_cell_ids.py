#!/usr/bin/env python3
"""
03_make_cell_ids.py

Build a fine-grid raster of coarse GHSL cell IDs and a lookup table with:
- coarse row/col
- coarse cell center coordinates
- original GHSL value
- overlap fraction with AOI
- adjusted GHSL value = ghsl_value * overlap_fraction

Example:
python 03_make_cell_ids.py \
  --coarse ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --template ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --aoi ~/data/aux/bbox_accra_dissolve.gpkg \
  --out-raster ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --out-lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import xy
from rasterio.warp import reproject
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fine-grid GHSL cell IDs and lookup table with overlap fractions."
    )
    parser.add_argument("--coarse", required=True, help="Path to coarse GHSL raster")
    parser.add_argument(
        "--template",
        required=True,
        help="Path to fine-grid template raster (e.g. PCA embeddings raster)",
    )
    parser.add_argument(
        "--aoi",
        required=True,
        help="Path to AOI vector used for clipping (gpkg/shp/geojson)",
    )
    parser.add_argument(
        "--out-raster",
        required=True,
        help="Output fine-grid raster of projected coarse cell IDs",
    )
    parser.add_argument(
        "--out-lookup",
        required=True,
        help="Output CSV lookup table for coarse cells",
    )
    parser.add_argument(
        "--area-crs",
        default="EPSG:6933",
        help="Projected CRS used for overlap-area calculations (default: EPSG:6933)",
    )
    return parser.parse_args()


def build_id_grid(coarse_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign a unique integer ID to each valid coarse raster cell.

    Returns
    -------
    id_grid : 2D int32 array
        Raster-sized grid of cell IDs, 0 for nodata.
    ids : 1D int64 array
        Cell IDs for valid cells.
    rows : 1D int64 array
        Row indices of valid coarse cells.
    cols : 1D int64 array
        Column indices of valid coarse cells.
    values : 1D float64 array
        GHSL values for valid coarse cells.
    """
    valid = np.isfinite(coarse_arr)
    rows, cols = np.where(valid)
    values = coarse_arr[valid].astype(np.float64, copy=False)

    ids = np.arange(1, rows.size + 1, dtype=np.int64)

    id_grid = np.zeros(coarse_arr.shape, dtype=np.int32)
    id_grid[rows, cols] = ids.astype(np.int32)

    return id_grid, ids, rows, cols, values


def compute_overlap_fractions(
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
    coarse_crs: CRS,
    aoi_path: Path,
    area_crs: str,
) -> np.ndarray:
    """
    Compute fraction of each coarse cell area overlapping the AOI polygon.

    overlap_fraction = area(cell ∩ AOI) / area(cell)

    Returns
    -------
    frac : 1D float64 array in [0, 1]
    """
    geoms = []
    for row, col in zip(rows, cols):
        left, bottom, right, top = window_bounds(
            Window(int(col), int(row), 1, 1), transform
        )
        geoms.append(box(left, bottom, right, top))

    cells_gdf = gpd.GeoDataFrame(
        {"cell_id": ids.astype(np.int64)},
        geometry=geoms,
        crs=coarse_crs,
    )

    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")

    # Dissolve into one AOI geometry
    aoi_union = aoi.dissolve()

    # Reproject both to an area-preserving / projected CRS
    cells_area = cells_gdf.to_crs(area_crs)
    aoi_area = aoi_union.to_crs(area_crs)

    aoi_geom = aoi_area.geometry.iloc[0]

    inter_area = cells_area.geometry.intersection(aoi_geom).area.to_numpy()
    cell_area = cells_area.geometry.area.to_numpy()

    frac = np.divide(
        inter_area,
        cell_area,
        out=np.zeros_like(inter_area, dtype=np.float64),
        where=cell_area > 0,
    )
    frac = np.clip(frac, 0.0, 1.0)

    return frac


def project_ids_to_template(
    id_grid: np.ndarray,
    coarse_transform,
    coarse_crs: CRS,
    template_profile: dict,
) -> np.ndarray:
    """
    Project coarse cell IDs onto the fine template grid with nearest-neighbor.
    """
    dst = np.zeros(
        (template_profile["height"], template_profile["width"]),
        dtype=np.int32,
    )

    reproject(
        source=id_grid,
        destination=dst,
        src_transform=coarse_transform,
        src_crs=coarse_crs,
        dst_transform=template_profile["transform"],
        dst_crs=template_profile["crs"],
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    return dst


def write_id_raster(
    out_path: Path,
    id_fine: np.ndarray,
    template_profile: dict,
) -> None:
    """
    Write fine-grid coarse-cell-ID raster.
    """
    profile = template_profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    profile.update(
        driver="GTiff",
        count=1,
        dtype="int32",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(id_fine, 1)


def write_lookup_csv(
    csv_path: Path,
    ids: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    overlap_fraction: np.ndarray,
    transform,
    crs: CRS,
) -> None:
    """
    Write lookup table with coarse-cell metadata.
    """
    xs, ys = xy(transform, rows, cols, offset="center")
    crs_wkt = crs.to_wkt() if crs is not None else ""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cell_id",
            "row",
            "col",
            "x_center",
            "y_center",
            "ghsl_value",
            "overlap_fraction",
            "ghsl_value_adj",
            "crs_wkt",
        ])

        for cell_id, row, col, x, y, value, frac in zip(
            ids, rows, cols, xs, ys, values, overlap_fraction
        ):
            writer.writerow([
                int(cell_id),
                int(row),
                int(col),
                float(x),
                float(y),
                float(value),
                float(frac),
                float(value * frac),
                crs_wkt,
            ])


def main() -> None:
    args = parse_args()

    coarse_path = Path(args.coarse).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()
    aoi_path = Path(args.aoi).expanduser().resolve()
    out_raster = Path(args.out_raster).expanduser().resolve()
    out_lookup = Path(args.out_lookup).expanduser().resolve()

    with rasterio.open(coarse_path) as coarse_src:
        coarse_arr = coarse_src.read(1).astype(np.float64, copy=False)
        coarse_transform = coarse_src.transform
        coarse_crs = coarse_src.crs
        coarse_height = coarse_src.height
        coarse_width = coarse_src.width

    with rasterio.open(template_path) as template_src:
        template_profile = template_src.profile.copy()
        template_transform = template_src.transform
        template_crs = template_src.crs
        template_height = template_src.height
        template_width = template_src.width

    print(f"[INFO] Coarse raster: {coarse_path}")
    print(f"[INFO] Coarse shape: {coarse_height} x {coarse_width}")
    print(f"[INFO] Template raster: {template_path}")
    print(f"[INFO] Template shape: {template_height} x {template_width}")

    id_grid, ids, rows, cols, values = build_id_grid(coarse_arr)
    print(f"[INFO] Valid coarse cells: {ids.size:,}")

    overlap_fraction = compute_overlap_fractions(
        ids=ids,
        rows=rows,
        cols=cols,
        transform=coarse_transform,
        coarse_crs=coarse_crs,
        aoi_path=aoi_path,
        area_crs=args.area_crs,
    )

    print(f"[INFO] Mean overlap fraction: {overlap_fraction.mean():.4f}")
    print(f"[INFO] Partial-overlap cells (<0.999): {(overlap_fraction < 0.999).sum():,}")

    id_fine = project_ids_to_template(
        id_grid=id_grid,
        coarse_transform=coarse_transform,
        coarse_crs=coarse_crs,
        template_profile=template_profile,
    )

    valid_fine = int((id_fine > 0).sum())
    unique_ids_fine = int(np.unique(id_fine[id_fine > 0]).size)

    print(f"[INFO] Valid fine pixels with cell IDs: {valid_fine:,}")
    print(f"[INFO] Unique coarse cells represented on fine grid: {unique_ids_fine:,}")

    write_id_raster(out_raster, id_fine, template_profile)
    print(f"[INFO] Saved cell ID raster to: {out_raster}")

    write_lookup_csv(
        csv_path=out_lookup,
        ids=ids,
        rows=rows,
        cols=cols,
        values=values,
        overlap_fraction=overlap_fraction,
        transform=coarse_transform,
        crs=coarse_crs,
    )
    print(f"[INFO] Saved lookup CSV to: {out_lookup}")

    positive = values > 0
    print(f"[INFO] Positive GHSL cells: {positive.sum():,}")
    print(f"[INFO] Total GHSL mass (raw): {values[positive].sum():,.4f}")
    print(f"[INFO] Total GHSL mass (adjusted): {(values[positive] * overlap_fraction[positive]).sum():,.4f}")


if __name__ == "__main__":
    main()