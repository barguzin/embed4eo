#!/usr/bin/env python3
"""
Compare two prediction rasters against a reference raster.

This is a compact diagnostic for old-vs-new baseline comparisons. It reports
MAE, RMSE, bias, Pearson correlation, and a global masked SSIM-style score for
each prediction against the same reference.

Example
-------
python scripts/14_compare_two_predictions_metrics.py \
  --reference ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --predictions ~/data/outputs/embed_wsf_norm.tif \
                ~/data/outputs/embed_wsf_diffnorm_norm.tif \
  --names embed_wsf embed_wsf_diffnorm \
  --output-csv ~/data/outputs/old_vs_new_wsf_metrics.csv

The first prediction defines the metric grid, matching the convention used by
09_evaluate_against_ghsl10m.py. The reference is aligned to that grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two prediction rasters against one reference raster.")
    p.add_argument("--reference", required=True, help="Reference raster used as the target for metrics.")
    p.add_argument("--predictions", nargs=2, required=True, help="Exactly two prediction rasters to compare.")
    p.add_argument("--names", nargs=2, default=None, help="Optional names for the two predictions.")
    p.add_argument("--output-csv", required=True, help="Output CSV path.")
    p.add_argument(
        "--resampling",
        choices=["nearest", "bilinear"],
        default="bilinear",
        help=(
            "Resampling used if the reference or second prediction is not on the first prediction grid. "
            "Default: bilinear."
        ),
    )
    return p.parse_args()


def read_single_band(path: Path) -> Tuple[np.ndarray, dict]:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def same_grid(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in ["height", "width", "crs", "transform"])


def align_to_target(src_path: Path, target_profile: dict, resampling_name: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    resampling = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear}[resampling_name]
    with rasterio.open(src_path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        dst = np.full((target_profile["height"], target_profile["width"]), np.nan, dtype=np.float32)
        src_nodata = src.nodata if src.nodata is not None else np.nan
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            src_nodata=src_nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


def load_prediction(path: Path, target_profile: dict, resampling_name: str, is_template: bool) -> Tuple[np.ndarray, dict]:
    arr, profile = read_single_band(path)
    if is_template or same_grid(profile, target_profile):
        return arr, profile
    print(f"[WARN] Prediction grid differs from first prediction; reprojecting with {resampling_name}: {path}")
    return align_to_target(path, target_profile, resampling_name), target_profile


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = np.sqrt(np.sum(x0**2) * np.sum(y0**2))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.sum(x0 * y0) / denom)


def global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """
    Global SSIM over valid pixels.

    This uses the standard SSIM formula over the vector of common valid pixels
    rather than a sliding image window, which keeps the metric well-defined for
    rasters with masks/nodata.
    """
    if x.size < 2:
        return np.nan
    data_min = min(float(np.min(x)), float(np.min(y)))
    data_max = max(float(np.max(x)), float(np.max(y)))
    data_range = data_max - data_min
    if data_range <= 0 or not np.isfinite(data_range):
        return np.nan

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    covxy = float(np.mean((x - mux) * (y - muy)))
    numerator = (2.0 * mux * muy + c1) * (2.0 * covxy + c2)
    denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return float(numerator / denominator)


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(ref)
    p = pred[valid].astype(np.float64, copy=False)
    r = ref[valid].astype(np.float64, copy=False)
    err = p - r
    return {
        "n_valid": int(p.size),
        "mae": float(np.mean(np.abs(err))) if p.size else np.nan,
        "rmse": float(np.sqrt(np.mean(err**2))) if p.size else np.nan,
        "bias": float(np.mean(err)) if p.size else np.nan,
        "pearson": pearson(p, r),
        "ssim": global_ssim(p, r),
    }


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference).expanduser().resolve()
    prediction_paths = [Path(p).expanduser().resolve() for p in args.predictions]
    names: List[str] = args.names or [p.stem for p in prediction_paths]
    output_csv = Path(args.output_csv).expanduser().resolve()

    template_prediction, template_profile = read_single_band(prediction_paths[0])
    reference_arr, reference_profile = read_single_band(reference_path)
    if same_grid(reference_profile, template_profile):
        reference = reference_arr
    else:
        print(f"[INFO] Reference grid differs from first prediction; reprojecting with {args.resampling}: {reference_path}")
        reference = align_to_target(reference_path, template_profile, args.resampling)

    rows = []
    for i, (name, pred_path) in enumerate(zip(names, prediction_paths)):
        if i == 0:
            pred = template_prediction
        else:
            pred, _ = load_prediction(pred_path, template_profile, args.resampling, is_template=False)
        metrics = compute_metrics(pred, reference)
        rows.append(
            {
                "name": name,
                "prediction": str(pred_path),
                "reference": str(reference_path),
                **metrics,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "prediction", "reference", "n_valid", "mae", "rmse", "bias", "pearson", "ssim"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved comparison metrics to: {output_csv}")
    for row in rows:
        print(
            f"[INFO] {row['name']}: "
            f"MAE={row['mae']:.6g}, RMSE={row['rmse']:.6g}, bias={row['bias']:.6g}, "
            f"Pearson={row['pearson']:.6g}, SSIM={row['ssim']:.6g}"
        )


if __name__ == "__main__":
    main()
