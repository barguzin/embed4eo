#!/usr/bin/env python3
"""
Evaluate downscaled built-up predictions against the WorldPop VIIRS FVF 2019
100 m covariate.

VIIRS is continuous but not in the same units as predicted built-up surface.
This evaluator therefore avoids same-unit error metrics. It aggregates each
prediction raster to the VIIRS 100 m grid using sum resampling, then reports:

- Pearson correlation between log1p(predicted built-up mass) and log1p(VIIRS)
- top-k overlap between highest-predicted cells and highest-VIIRS cells

Example
-------
python scripts/10_evaluate_against_viirs.py \
  --predictions ~/data/outputs/wsf_uniform_baseline.tif \
                ~/data/outputs/embed_only_norm.tif \
                ~/data/outputs/embed_wsf_norm.tif \
  --names wsf_uniform embed_only embed_wsf \
  --viirs ~/data/VIIRS/cropped_viirs_fvf_2019_100m.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --output-csv ~/data/outputs/evaluation_viirs_metrics.csv \
  --output-fig ~/data/outputs/evaluation_viirs_summary.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_TOPK_FRACS = [0.01, 0.05, 0.10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate predictions against WorldPop VIIRS FVF by aggregating predictions to VIIRS grid."
    )
    p.add_argument("--predictions", nargs="+", required=True, help="Prediction rasters to evaluate.")
    p.add_argument("--names", nargs="+", default=None, help="Model names, one per prediction raster.")
    p.add_argument("--viirs", required=True, help="Prepared VIIRS FVF raster, used as the target grid.")
    p.add_argument(
        "--cell-ids",
        default=None,
        help="Optional fine-grid cell-ID raster. Used only to mask predictions before aggregation.",
    )
    p.add_argument(
        "--topk-fracs",
        nargs="+",
        type=float,
        default=DEFAULT_TOPK_FRACS,
        help="Top-k cell fractions for overlap diagnostics. Default: 0.01 0.05 0.10.",
    )
    p.add_argument("--output-csv", required=True, help="Output long-format metrics CSV.")
    p.add_argument("--output-fig", default=None, help="Optional PNG summary figure.")
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


def require_same_grid(candidate: dict, template: dict, candidate_name: str, template_name: str) -> None:
    if same_grid(candidate, template):
        return
    details = []
    for key in ["height", "width", "crs", "transform"]:
        if candidate[key] != template[key]:
            details.append(f"{key}: {candidate[key]} != {template[key]}")
    raise ValueError(f"{candidate_name} and {template_name} grids do not match. " + "; ".join(details))


def aggregate_prediction_to_viirs_grid(
    pred_path: Path,
    viirs_profile: dict,
    cell_ids_path: Optional[Path] = None,
) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(pred_path) as src:
        pred = src.read(1).astype(np.float32, copy=False)
        pred_nodata = src.nodata if src.nodata is not None else np.nan
        valid = np.isfinite(pred)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= pred != src.nodata

        if cell_ids_path is not None:
            with rasterio.open(cell_ids_path) as ids_src:
                require_same_grid(ids_src.profile, src.profile, str(cell_ids_path), str(pred_path))
                ids = ids_src.read(1)
                valid &= np.isfinite(ids) & (ids > 0)

        pred_masked = np.where(valid, pred, 0.0).astype(np.float32, copy=False)
        dst = np.zeros((viirs_profile["height"], viirs_profile["width"]), dtype=np.float32)
        reproject(
            source=pred_masked,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=viirs_profile["transform"],
            dst_crs=viirs_profile["crs"],
            src_nodata=0.0,
            dst_nodata=0.0,
            resampling=Resampling.sum,
        )
    return dst


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 and np.isfinite(den) else np.nan


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def top_fraction_mask(values: np.ndarray, fraction: float) -> Tuple[np.ndarray, float]:
    if not 0 < fraction < 1:
        raise ValueError("--topk-fracs values must be in (0, 1)")
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool), np.nan
    n_top = max(1, int(np.ceil(values.size * fraction)))
    cutoff = float(np.partition(values, values.size - n_top)[values.size - n_top])
    return values >= cutoff, cutoff


def add_row(rows: List[dict], model: str, metric_group: str, metric: str, value: float, **extra) -> None:
    row = {
        "model": model,
        "metric_group": metric_group,
        "metric": metric,
        "value": value,
        "topk_frac": None,
    }
    row.update(extra)
    rows.append(row)


def compute_metrics(
    pred_100m: np.ndarray,
    viirs: np.ndarray,
    topk_fracs: Iterable[float],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    valid = np.isfinite(pred_100m) & np.isfinite(viirs) & (pred_100m >= 0) & (viirs >= 0)
    pred = pred_100m[valid].astype(np.float64, copy=False)
    viirs_valid = viirs[valid].astype(np.float64, copy=False)

    pred_log = np.log1p(pred)
    viirs_log = np.log1p(viirs_valid)

    total_pred_mass = float(np.sum(pred))
    total_viirs = float(np.sum(viirs_valid))
    base_metrics = {
        "valid_cell_count": float(pred.size),
        "prediction_aggregation": "sum_to_viirs_grid",
        "pred_mass_total_100m": total_pred_mass,
        "viirs_total": total_viirs,
        "pearson_log1p": pearson(pred_log, viirs_log),
        "pred_log1p_mean": float(np.mean(pred_log)) if pred.size else np.nan,
        "viirs_log1p_mean": float(np.mean(viirs_log)) if viirs_valid.size else np.nan,
    }

    topk: Dict[str, Dict[str, float]] = {}
    for frac in topk_fracs:
        pred_top, pred_threshold = top_fraction_mask(pred, frac)
        viirs_top, viirs_threshold = top_fraction_mask(viirs_valid, frac)
        overlap = pred_top & viirs_top
        pred_top_mass = float(np.sum(pred[pred_top]))
        viirs_top_total = float(np.sum(viirs_valid[viirs_top]))
        overlap_pred_mass = float(np.sum(pred[overlap]))
        overlap_viirs_total = float(np.sum(viirs_valid[overlap]))

        topk[str(frac)] = {
            "topk_frac": float(frac),
            "pred_topk_threshold": pred_threshold,
            "viirs_topk_threshold": viirs_threshold,
            "topk_cell_count": float(np.sum(pred_top)),
            "topk_overlap_cell_count": float(np.sum(overlap)),
            "topk_overlap_share_of_pred_topk": safe_div(float(np.sum(overlap)), float(np.sum(pred_top))),
            "topk_overlap_share_of_viirs_topk": safe_div(float(np.sum(overlap)), float(np.sum(viirs_top))),
            "topk_overlap_lift": safe_div(safe_div(float(np.sum(overlap)), float(np.sum(pred_top))), frac),
            "topk_pred_mass_share": safe_div(pred_top_mass, total_pred_mass),
            "topk_viirs_share": safe_div(viirs_top_total, total_viirs),
            "topk_overlap_pred_mass_share": safe_div(overlap_pred_mass, pred_top_mass),
            "topk_overlap_viirs_share": safe_div(overlap_viirs_total, viirs_top_total),
        }

    return base_metrics, topk


def write_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = ["model", "metric_group", "metric", "value", "topk_frac"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_from_rows(rows: List[dict], model: str, metric_group: str, metric: str, topk_frac=None) -> Optional[float]:
    for row in rows:
        if row["model"] != model or row["metric_group"] != metric_group or row["metric"] != metric:
            continue
        if topk_frac is not None and row.get("topk_frac") != topk_frac:
            continue
        return row["value"]
    return None


def write_figure(rows: List[dict], models: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    correlations = [
        value_from_rows(rows, model, "correlation", "pearson_log1p") or 0.0
        for model in models
    ]
    axes[0].bar(models, correlations)
    axes[0].set_title("VIIRS agreement: Pearson on log1p values")
    axes[0].set_ylabel("Pearson correlation")
    axes[0].set_ylim(-1, 1)
    axes[0].tick_params(axis="x", rotation=25)

    top_rows = [
        r for r in rows
        if r["metric_group"] == "topk" and r["metric"] == "topk_overlap_share_of_pred_topk"
    ]
    fracs = sorted({float(r["topk_frac"]) for r in top_rows if r["topk_frac"] is not None})
    x = np.arange(len(fracs))
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        vals = [
            value_from_rows(rows, model, "topk", "topk_overlap_share_of_pred_topk", topk_frac=frac) or 0.0
            for frac in fracs
        ]
        axes[1].bar(x + i * width, vals, width=width, label=model)
    axes[1].set_title("Top-k overlap with highest VIIRS cells")
    axes[1].set_ylabel("Share of prediction top-k also in VIIRS top-k")
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    axes[1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if any(not 0 < f < 1 for f in args.topk_fracs):
        raise ValueError("--topk-fracs values must be in (0, 1)")

    pred_paths = [Path(p).expanduser().resolve() for p in args.predictions]
    names = args.names if args.names is not None else [p.stem for p in pred_paths]
    if len(names) != len(pred_paths):
        raise ValueError("--names must have the same length as --predictions")

    viirs_path = Path(args.viirs).expanduser().resolve()
    viirs, viirs_profile = read_single_band(viirs_path)
    cell_ids_path = Path(args.cell_ids).expanduser().resolve() if args.cell_ids else None

    rows: List[dict] = []
    for model, pred_path in zip(names, pred_paths):
        print(f"[INFO] Evaluating {model}")
        pred_100m = aggregate_prediction_to_viirs_grid(pred_path, viirs_profile, cell_ids_path)
        base_metrics, topk = compute_metrics(pred_100m, viirs, args.topk_fracs)
        for metric, value in base_metrics.items():
            group = "correlation" if metric == "pearson_log1p" else "summary"
            add_row(rows, model, group, metric, value)
        for frac_key, metrics in topk.items():
            frac = float(frac_key)
            for metric, value in metrics.items():
                add_row(rows, model, "topk", metric, value, topk_frac=frac)

    out_csv = Path(args.output_csv).expanduser().resolve()
    write_csv(rows, out_csv)
    print(f"[INFO] Saved VIIRS metrics CSV to: {out_csv}")

    if args.output_fig:
        out_fig = Path(args.output_fig).expanduser().resolve()
        write_figure(rows, names, out_fig)
        print(f"[INFO] Saved VIIRS summary figure to: {out_fig}")

    print("[INFO] Prediction rasters were aggregated to the VIIRS 100 m grid using sum resampling.")


if __name__ == "__main__":
    main()
