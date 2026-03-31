#!/usr/bin/env python3
"""Mass-preserving baseline: allocate each GHSL coarse value uniformly across
WSF-positive fine pixels within that coarse cell.

If a coarse cell contains no WSF-positive fine pixels, the script falls back to
uniform allocation across all valid fine pixels in that cell.

Inputs
------
--wsf         Fine-grid WSF binary raster aligned to the template grid.
--cell-ids    Fine-grid raster mapping each fine pixel to its parent coarse cell.
--lookup      CSV produced by 03_make_cell_ids.py; must include `cell_id` and
              `ghsl_value`.
--output      Output fine-grid float32 raster of downscaled built-up surface.

Optional outputs
----------------
--report      JSON with summary diagnostics.
--fallback    Raster showing which fine pixels belong to cells that required
              fallback allocation (1=fallback cell, 0=normal cell, nodata elsewhere).

Example
-------
python 04_baseline_wsf_uniform.py \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --output ~/data/outputs/wsf_uniform_baseline.tif \
  --report ~/data/outputs/wsf_uniform_baseline_report.json \
  --fallback ~/data/outputs/wsf_uniform_baseline_fallback_cells.tif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


NODATA_ID = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsf", required=True, help="Path to aligned fine-grid WSF binary raster")
    parser.add_argument("--cell-ids", required=True, help="Path to fine-grid coarse-cell ID raster")
    parser.add_argument("--lookup", required=True, help="Path to coarse-cell lookup CSV")
    parser.add_argument("--output", required=True, help="Path to output fine-grid baseline raster")
    parser.add_argument("--report", required=False, help="Optional JSON report path")
    parser.add_argument("--fallback", required=False, help="Optional raster marking fallback cells")
    parser.add_argument(
        "--value-column",
        default="ghsl_value",
        help="Column in the lookup CSV holding the coarse GHSL value (default: ghsl_value)",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="WSF values greater than this threshold are treated as built (default: 0.0)",
    )
    return parser.parse_args()


def load_lookup(lookup_path: Path, value_column: str) -> tuple[np.ndarray, dict[int, float]]:
    df = pd.read_csv(lookup_path)
    required = {"cell_id", value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Lookup CSV is missing required columns: {sorted(missing)}")

    ids = df["cell_id"].to_numpy(dtype=np.int64)
    vals = df[value_column].to_numpy(dtype=np.float64)
    return ids, dict(zip(ids.tolist(), vals.tolist()))


def main() -> None:
    args = parse_args()
    wsf_path = Path(args.wsf).expanduser().resolve()
    cell_ids_path = Path(args.cell_ids).expanduser().resolve()
    lookup_path = Path(args.lookup).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else None
    fallback_path = Path(args.fallback).expanduser().resolve() if args.fallback else None

    coarse_ids, coarse_value_map = load_lookup(lookup_path, args.value_column)

    with rasterio.open(cell_ids_path) as ids_src, rasterio.open(wsf_path) as wsf_src:
        if (
            ids_src.width != wsf_src.width
            or ids_src.height != wsf_src.height
            or ids_src.transform != wsf_src.transform
            or ids_src.crs != wsf_src.crs
        ):
            raise ValueError("WSF raster and cell-ID raster must be perfectly aligned")

        cell_ids = ids_src.read(1, masked=True)
        ids_data = np.asarray(cell_ids.filled(NODATA_ID), dtype=np.int64)
        fine_valid = (~cell_ids.mask) & (ids_data != NODATA_ID)

        wsf = wsf_src.read(1, masked=True)
        wsf_data = np.asarray(wsf.filled(np.nan), dtype=np.float32)
        wsf_positive = (~wsf.mask) & np.isfinite(wsf_data) & (wsf_data > args.positive_threshold)

        out = np.full(ids_data.shape, np.nan, dtype=np.float32)
        fallback_mask = np.full(ids_data.shape, 0, dtype=np.uint8)

        unique_ids = np.unique(ids_data[fine_valid])
        unique_ids = unique_ids[unique_ids != NODATA_ID]

        print(f"[INFO] Fine raster shape: {ids_src.height} x {ids_src.width}")
        print(f"[INFO] Valid fine pixels with cell IDs: {fine_valid.sum():,}")
        print(f"[INFO] Unique coarse cells represented on fine grid: {unique_ids.size:,}")

        fallback_cells = 0
        missing_lookup_cells = 0
        zero_or_negative_cells = 0

        for cell_id in unique_ids:
            coarse_value = coarse_value_map.get(int(cell_id))
            if coarse_value is None:
                missing_lookup_cells += 1
                continue

            cell_mask = fine_valid & (ids_data == cell_id)
            n_valid = int(cell_mask.sum())
            if n_valid == 0:
                continue

            if not np.isfinite(coarse_value) or coarse_value <= 0:
                zero_or_negative_cells += 1
                out[cell_mask] = 0.0
                continue

            built_mask = cell_mask & wsf_positive
            n_built = int(built_mask.sum())

            if n_built > 0:
                out[built_mask] = np.float32(coarse_value / n_built)
                out[cell_mask & (~wsf_positive)] = 0.0
            else:
                fallback_cells += 1
                fallback_mask[cell_mask] = 1
                out[cell_mask] = np.float32(coarse_value / n_valid)

        out_profile = ids_src.profile.copy()
        out_profile.pop("blockxsize", None)
        out_profile.pop("blockysize", None)
        out_profile.update(
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(out, 1)

        if fallback_path is not None:
            fb_profile = ids_src.profile.copy()
            fb_profile.pop("blockxsize", None)
            fb_profile.pop("blockysize", None)
            fb_profile.update(
                count=1,
                dtype="uint8",
                nodata=255,
                compress="deflate",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF="IF_SAFER",
            )
            fb = np.full(ids_data.shape, 255, dtype=np.uint8)
            fb[fine_valid] = fallback_mask[fine_valid]
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            if fallback_path.exists():
                fallback_path.unlink()
            with rasterio.open(fallback_path, "w", **fb_profile) as dst:
                dst.write(fb, 1)

        allocated_total = float(np.nansum(out))
        expected_total = float(
            sum(v for v in coarse_value_map.values() if np.isfinite(v) and v > 0)
        )
        in_wsf_total = float(np.nansum(np.where(wsf_positive, out, 0.0)))
        out_wsf_total = float(np.nansum(np.where(~wsf_positive, np.nan_to_num(out, nan=0.0), 0.0)))
        frac_inside_wsf = in_wsf_total / allocated_total if allocated_total > 0 else np.nan

        report = {
            "wsf_path": str(wsf_path),
            "cell_ids_path": str(cell_ids_path),
            "lookup_path": str(lookup_path),
            "output_path": str(out_path),
            "value_column": args.value_column,
            "positive_threshold": args.positive_threshold,
            "n_fine_pixels_valid": int(fine_valid.sum()),
            "n_coarse_cells_on_fine_grid": int(unique_ids.size),
            "n_fallback_cells": int(fallback_cells),
            "n_missing_lookup_cells": int(missing_lookup_cells),
            "n_zero_or_negative_cells": int(zero_or_negative_cells),
            "allocated_total": allocated_total,
            "expected_total_positive_cells": expected_total,
            "mass_difference": allocated_total - expected_total,
            "mass_fraction_inside_wsf": frac_inside_wsf,
            "mass_fraction_outside_wsf": (out_wsf_total / allocated_total) if allocated_total > 0 else np.nan,
        }

    print(f"[INFO] Saved baseline raster to: {out_path}")
    print(f"[INFO] Fallback coarse cells: {fallback_cells:,}")
    print(f"[INFO] Allocated total: {allocated_total:,.4f}")
    print(f"[INFO] Expected total over positive coarse cells: {expected_total:,.4f}")
    print(f"[INFO] Mass fraction inside WSF: {frac_inside_wsf:.4f}")

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[INFO] Saved report to: {report_path}")


if __name__ == "__main__":
    main()
