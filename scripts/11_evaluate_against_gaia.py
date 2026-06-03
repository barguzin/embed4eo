#!/usr/bin/env python3
"""
Evaluate downscaled built-up predictions against GAIA 2019 impervious extent.

GAIA is used here as a binary impervious/non-impervious validation layer. The
prepared GAIA raster is 30 m, so each fine prediction raster is aggregated to
the GAIA grid using sum resampling before metrics are computed.

Example
-------
python scripts/11_evaluate_against_gaia.py \
  --predictions ~/data/outputs/wsf_uniform_baseline.tif \
                ~/data/outputs/embed_only_norm.tif \
                ~/data/outputs/embed_wsf_norm.tif \
  --names wsf_uniform embed_only embed_wsf \
  --gaia-impervious ~/data/GAIA/cropped_gaia_impervious_2019.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --output-csv ~/data/outputs/evaluation_gaia_metrics.csv \
  --output-fig ~/data/outputs/evaluation_gaia_summary.png \
  --output-map-dir ~/data/outputs/evaluation_gaia_maps
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_TOPK_FRACS = [0.01, 0.05, 0.10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate predictions against binary GAIA 2019 impervious extent."
    )
    p.add_argument("--predictions", nargs="+", required=True, help="Prediction rasters to evaluate.")
    p.add_argument("--names", nargs="+", default=None, help="Model names, one per prediction raster.")
    p.add_argument("--gaia-impervious", required=True, help="Prepared binary GAIA 2019 impervious raster.")
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
    p.add_argument("--output-map-dir", default=None, help="Optional directory for leakage GeoTIFF maps.")
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


def aggregate_prediction_to_gaia_grid(
    pred_path: Path,
    gaia_profile: dict,
    cell_ids_path: Optional[Path] = None,
) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(pred_path) as src:
        pred = src.read(1).astype(np.float32, copy=False)
        valid = np.isfinite(pred)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= pred != src.nodata

        if cell_ids_path is not None:
            with rasterio.open(cell_ids_path) as ids_src:
                require_same_grid(ids_src.profile, src.profile, str(cell_ids_path), str(pred_path))
                ids = ids_src.read(1)
                valid &= np.isfinite(ids) & (ids > 0)

        pred_masked = np.where(valid, pred, 0.0).astype(np.float32, copy=False)
        dst = np.zeros((gaia_profile["height"], gaia_profile["width"]), dtype=np.float32)
        reproject(
            source=pred_masked,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=gaia_profile["transform"],
            dst_crs=gaia_profile["crs"],
            src_nodata=0.0,
            dst_nodata=0.0,
            resampling=Resampling.sum,
        )
    return dst


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 and np.isfinite(den) else np.nan


def top_fraction_mask(values: np.ndarray, fraction: float) -> Tuple[np.ndarray, float]:
    if not 0 < fraction < 1:
        raise ValueError("--topk-fracs values must be in (0, 1)")
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool), np.nan
    n_top = max(1, int(np.ceil(values.size * fraction)))
    cutoff = float(np.partition(values, values.size - n_top)[values.size - n_top])
    return values >= cutoff, cutoff


def prevalence_matched_metrics(pred: np.ndarray, reference: np.ndarray) -> dict:
    prevalence = float(np.mean(reference)) if reference.size else np.nan
    if not np.isfinite(prevalence) or prevalence <= 0 or prevalence >= 1:
        pred_positive = pred > 0
        threshold = 0.0
    else:
        pred_positive, threshold = top_fraction_mask(pred, prevalence)

    ref_positive = reference.astype(bool)
    tp = float(np.sum(pred_positive & ref_positive))
    fp = float(np.sum(pred_positive & ~ref_positive))
    fn = float(np.sum(~pred_positive & ref_positive))
    tn = float(np.sum(~pred_positive & ~ref_positive))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "threshold": threshold,
        "pred_positive_share": safe_div(float(np.sum(pred_positive)), float(pred.size)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "iou": safe_div(tp, tp + fp + fn),
        "balanced_accuracy": np.nanmean([recall, specificity]),
    }


def add_row(
    rows: List[dict],
    model: str,
    metric_group: str,
    metric: str,
    value,
    topk_frac: Optional[float] = None,
) -> None:
    rows.append(
        {
            "model": model,
            "metric_group": metric_group,
            "metric": metric,
            "value": value,
            "topk_frac": topk_frac,
        }
    )


def compute_metrics(pred_30m: np.ndarray, gaia: np.ndarray, topk_fracs: Iterable[float]) -> List[dict]:
    valid = np.isfinite(pred_30m) & np.isfinite(gaia) & (pred_30m >= 0) & np.isin(gaia, [0, 1])
    pred = pred_30m[valid].astype(np.float64, copy=False)
    gaia_valid = gaia[valid].astype(np.uint8, copy=False)
    gaia_built = gaia_valid == 1

    total_mass = float(np.sum(pred))
    mass_in_built = float(np.sum(pred[gaia_built]))
    mass_outside_built = float(np.sum(pred[~gaia_built]))
    prevalence = float(np.mean(gaia_built)) if gaia_built.size else np.nan

    rows: List[dict] = []
    summary = {
        "valid_cell_count": float(pred.size),
        "prediction_aggregation": "sum_to_gaia_30m_grid",
        "gaia_impervious_prevalence": prevalence,
        "gaia_impervious_cell_count": float(np.sum(gaia_built)),
        "gaia_nonimpervious_cell_count": float(np.sum(~gaia_built)),
        "pred_mass_total_30m": total_mass,
        "pred_mass_in_gaia_impervious": mass_in_built,
        "pred_mass_outside_gaia_impervious": mass_outside_built,
        "share_mass_in_gaia_impervious": safe_div(mass_in_built, total_mass),
        "share_mass_outside_gaia_impervious": safe_div(mass_outside_built, total_mass),
        "mean_pred_in_gaia_impervious": float(np.mean(pred[gaia_built])) if np.any(gaia_built) else np.nan,
        "mean_pred_outside_gaia_impervious": float(np.mean(pred[~gaia_built])) if np.any(~gaia_built) else np.nan,
        "median_pred_in_gaia_impervious": float(np.median(pred[gaia_built])) if np.any(gaia_built) else np.nan,
        "median_pred_outside_gaia_impervious": float(np.median(pred[~gaia_built])) if np.any(~gaia_built) else np.nan,
    }
    summary["mean_pred_impervious_to_nonimpervious_ratio"] = safe_div(
        summary["mean_pred_in_gaia_impervious"],
        summary["mean_pred_outside_gaia_impervious"],
    )
    for metric, value in summary.items():
        add_row(rows, "", "summary", metric, value)

    for frac in topk_fracs:
        pred_top, pred_threshold = top_fraction_mask(pred, frac)
        overlap_count = float(np.sum(pred_top & gaia_built))
        pred_top_count = float(np.sum(pred_top))
        pred_top_mass = float(np.sum(pred[pred_top]))
        overlap_mass = float(np.sum(pred[pred_top & gaia_built]))
        p_built_given_topk = safe_div(overlap_count, pred_top_count)
        topk_metrics = {
            "topk_pred_threshold": pred_threshold,
            "topk_cell_count": pred_top_count,
            "topk_gaia_impervious_cell_count": overlap_count,
            "topk_gaia_impervious_overlap": p_built_given_topk,
            "topk_gaia_impervious_lift": safe_div(p_built_given_topk, prevalence),
            "topk_pred_mass": pred_top_mass,
            "topk_pred_mass_share": safe_div(pred_top_mass, total_mass),
            "topk_pred_mass_in_gaia_impervious": overlap_mass,
            "topk_pred_mass_share_in_gaia_impervious": safe_div(overlap_mass, pred_top_mass),
        }
        for metric, value in topk_metrics.items():
            add_row(rows, "", "topk", metric, value, topk_frac=float(frac))

    for metric, value in prevalence_matched_metrics(pred, gaia_valid).items():
        add_row(rows, "", "prevalence_matched_binary", metric, value)

    return rows


def rows_for_model(model: str, metric_rows: List[dict]) -> List[dict]:
    rows = []
    for row in metric_rows:
        out = row.copy()
        out["model"] = model
        rows.append(out)
    return rows


def write_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = ["model", "metric_group", "metric", "value", "topk_frac"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_from_rows(
    rows: List[dict],
    model: str,
    metric_group: str,
    metric: str,
    topk_frac=None,
) -> Optional[float]:
    for row in rows:
        if row["model"] != model or row["metric_group"] != metric_group or row["metric"] != metric:
            continue
        if topk_frac is not None and row.get("topk_frac") != topk_frac:
            continue
        return row["value"]
    return None


def write_figure(rows: List[dict], models: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(len(models), dtype=float)

    share_vals = [
        value_from_rows(rows, model, "summary", "share_mass_in_gaia_impervious") or 0.0
        for model in models
    ]
    axes[0, 0].bar(x, share_vals)
    axes[0, 0].set_title("Predicted mass inside GAIA impervious")
    axes[0, 0].set_ylabel("Share of total predicted mass")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=25)

    outside_vals = [
        value_from_rows(rows, model, "summary", "pred_mass_outside_gaia_impervious") or 0.0
        for model in models
    ]
    axes[0, 1].bar(x, outside_vals)
    axes[0, 1].set_title("Absolute predicted mass outside GAIA impervious")
    axes[0, 1].set_ylabel("Predicted built-up surface")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=25)

    top_rows = [
        r for r in rows
        if r["metric_group"] == "topk" and r["metric"] == "topk_gaia_impervious_overlap"
    ]
    fracs = sorted({float(r["topk_frac"]) for r in top_rows if r["topk_frac"] is not None})
    x_top = np.arange(len(fracs), dtype=float)
    width = 0.8 / max(1, len(models))
    prevalence = value_from_rows(rows, models[0], "summary", "gaia_impervious_prevalence")
    if prevalence is not None and np.isfinite(float(prevalence)):
        axes[1, 0].axhline(float(prevalence), color="black", linestyle="--", linewidth=1, label="GAIA prevalence")
    for i, model in enumerate(models):
        vals = [
            value_from_rows(rows, model, "topk", "topk_gaia_impervious_overlap", topk_frac=frac) or 0.0
            for frac in fracs
        ]
        axes[1, 0].bar(x_top + i * width, vals, width=width, label=model)
    axes[1, 0].set_title("Top-k overlap with GAIA impervious")
    axes[1, 0].set_ylabel("P(GAIA impervious | top-k predicted)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticks(x_top + width * (len(models) - 1) / 2)
    axes[1, 0].set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    axes[1, 0].legend(fontsize=8)

    f1_vals = [
        value_from_rows(rows, model, "prevalence_matched_binary", "f1") or 0.0
        for model in models
    ]
    iou_vals = [
        value_from_rows(rows, model, "prevalence_matched_binary", "iou") or 0.0
        for model in models
    ]
    axes[1, 1].bar(x - 0.18, f1_vals, width=0.36, label="F1")
    axes[1, 1].bar(x + 0.18, iou_vals, width=0.36, label="IoU")
    axes[1, 1].set_title("Prevalence-matched binary agreement")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=25)
    axes[1, 1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_leakage_map(
    pred_30m: np.ndarray,
    gaia: np.ndarray,
    gaia_profile: dict,
    out_path: Path,
) -> None:
    import rasterio

    valid = np.isfinite(pred_30m) & np.isfinite(gaia) & np.isin(gaia, [0, 1])
    leakage = np.where(valid & (gaia == 0), pred_30m, 0.0).astype(np.float32)
    profile = gaia_profile.copy()
    profile.update(count=1, dtype="float32", nodata=0.0, compress="lzw")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(leakage, 1)


def main() -> None:
    args = parse_args()
    if any(not 0 < f < 1 for f in args.topk_fracs):
        raise ValueError("--topk-fracs values must be in (0, 1)")

    pred_paths = [Path(p).expanduser().resolve() for p in args.predictions]
    names = args.names if args.names is not None else [p.stem for p in pred_paths]
    if len(names) != len(pred_paths):
        raise ValueError("--names must have the same length as --predictions")

    gaia_path = Path(args.gaia_impervious).expanduser().resolve()
    gaia, gaia_profile = read_single_band(gaia_path)
    cell_ids_path = Path(args.cell_ids).expanduser().resolve() if args.cell_ids else None

    all_rows: List[dict] = []
    for model, pred_path in zip(names, pred_paths):
        print(f"[INFO] Evaluating {model}")
        pred_30m = aggregate_prediction_to_gaia_grid(pred_path, gaia_profile, cell_ids_path)
        all_rows.extend(rows_for_model(model, compute_metrics(pred_30m, gaia, args.topk_fracs)))

        if args.output_map_dir:
            map_dir = Path(args.output_map_dir).expanduser().resolve()
            write_leakage_map(
                pred_30m,
                gaia,
                gaia_profile,
                map_dir / f"{model}_outside_gaia_impervious_predicted_mass.tif",
            )

    out_csv = Path(args.output_csv).expanduser().resolve()
    write_csv(all_rows, out_csv)
    print(f"[INFO] Saved GAIA metrics CSV to: {out_csv}")

    if args.output_fig:
        out_fig = Path(args.output_fig).expanduser().resolve()
        write_figure(all_rows, names, out_fig)
        print(f"[INFO] Saved GAIA summary figure to: {out_fig}")

    if args.output_map_dir:
        print(f"[INFO] Saved GAIA leakage maps to: {Path(args.output_map_dir).expanduser().resolve()}")
    print("[INFO] Prediction rasters were aggregated to the GAIA 30 m grid using sum resampling.")


if __name__ == "__main__":
    main()
