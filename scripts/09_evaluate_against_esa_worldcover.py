#!/usr/bin/env python3
"""
Evaluate downscaled built-up predictions against ESA WorldCover 2020.

ESA WorldCover is categorical land cover, not a continuous built-up-surface
target. This evaluator therefore reports spatial plausibility and categorical
agreement diagnostics instead of same-unit MAE/RMSE errors.

Example
-------
python scripts/09_evaluate_against_esa_worldcover.py \
  --predictions ~/data/outputs/wsf_uniform_baseline.tif \
                ~/data/outputs/embed_only_norm.tif \
                ~/data/outputs/embed_wsf_norm.tif \
  --names wsf_uniform embed_only embed_wsf \
  --esa-worldcover ~/data/ESA_cover_2020/cropped_esa_worldcover_2020.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --built-class 50 \
  --output-csv ~/data/outputs/evaluation_esa_metrics.csv \
  --output-json ~/data/outputs/evaluation_esa_metrics.json \
  --output-fig ~/data/outputs/evaluation_esa_summary.png \
  --output-map-dir ~/data/outputs/evaluation_esa_maps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


ESA_CLASS_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
HARD_NONBUILT_CLASSES = {70, 80, 90, 95}
INTERPRETABLE_LEAKAGE_CLASSES = {
    30: "grassland",
    40: "cropland",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "water",
    90: "wetland",
    95: "mangroves",
}
DEFAULT_TOPK_FRACS = [0.01, 0.05, 0.10]
DEFAULT_MIN_PRED_THRESHOLDS = [0.0, 1.0, 5.0, 10.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate downscaled built-up predictions against categorical ESA WorldCover."
    )
    p.add_argument("--predictions", nargs="+", required=True, help="Prediction rasters to evaluate.")
    p.add_argument("--names", nargs="+", default=None, help="Model names, one per prediction raster.")
    p.add_argument("--esa-worldcover", required=True, help="Aligned ESA WorldCover categorical raster.")
    p.add_argument("--wsf", default=None, help="Optional aligned WSF raster for diagnostics.")
    p.add_argument(
        "--cell-ids",
        default=None,
        help="Optional aligned coarse-cell ID raster used only to restrict valid pixels.",
    )
    p.add_argument("--built-class", type=int, default=50, help="ESA built-up class code. Default: 50.")
    p.add_argument("--output-csv", required=True, help="Output long-format metrics CSV.")
    p.add_argument("--output-json", default=None, help="Optional metrics JSON.")
    p.add_argument("--output-fig", default=None, help="Optional summary figure PNG.")
    p.add_argument(
        "--output-map-dir",
        default=None,
        help="Optional directory for leakage rasters/maps, one set per model.",
    )
    p.add_argument(
        "--topk-fracs",
        nargs="+",
        type=float,
        default=DEFAULT_TOPK_FRACS,
        help="Top-prediction pixel fractions. Default: 0.01 0.05 0.10.",
    )
    p.add_argument(
        "--min-pred-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_MIN_PRED_THRESHOLDS,
        help="Fixed prediction thresholds for threshold-sensitive binary metrics.",
    )
    return p.parse_args()


def read_single_band(path: Path, categorical: bool = False) -> Tuple[np.ndarray, dict]:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    arr = arr.astype(np.float32 if not categorical else np.float64, copy=False)
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def same_grid(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in ["height", "width", "crs", "transform"])


def require_same_grid(
    candidate: dict,
    template: dict,
    candidate_name: str,
    template_name: str,
    esa_message: bool = False,
) -> None:
    if same_grid(candidate, template):
        return
    details = []
    for key in ["height", "width", "crs", "transform"]:
        if candidate[key] != template[key]:
            details.append(f"{key}: {candidate[key]} != {template[key]}")
    if esa_message:
        raise ValueError(
            "Prediction and ESA WorldCover grids do not match. "
            "Please crop/resample ESA WorldCover to the prediction grid before running evaluation. "
            + "; ".join(details)
        )
    raise ValueError(
        f"{candidate_name} and {template_name} grids do not match. "
        + "; ".join(details)
    )


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 and np.isfinite(den) else np.nan


def finite_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return np.nan
    return float(np.percentile(values, q))


def top_fraction_mask(values: np.ndarray, fraction: float) -> Tuple[np.ndarray, float]:
    if not 0 < fraction < 1:
        raise ValueError("--topk-fracs values must be in (0, 1)")
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool), np.nan
    n_top = max(1, int(np.ceil(values.size * fraction)))
    cutoff = float(np.partition(values, values.size - n_top)[values.size - n_top])
    return values >= cutoff, cutoff


def binary_agreement(pred_built: np.ndarray, esa_built: np.ndarray) -> Dict[str, float]:
    tp = float(np.sum(pred_built & esa_built))
    fp = float(np.sum(pred_built & ~esa_built))
    fn = float(np.sum(~pred_built & esa_built))
    tn = float(np.sum(~pred_built & ~esa_built))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    iou = safe_div(tp, tp + fp + fn)
    balanced_accuracy = np.nanmean([recall, specificity])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def add_row(rows: List[dict], model: str, metric_group: str, metric: str, value: float, **extra) -> None:
    row = {
        "model": model,
        "metric_group": metric_group,
        "metric": metric,
        "value": value,
        "esa_class": None,
        "esa_class_label": None,
        "topk_frac": None,
        "threshold": None,
        "group": None,
        "comparison_model": None,
    }
    row.update(extra)
    rows.append(row)


def add_metrics(rows: List[dict], model: str, metric_group: str, metrics: Dict[str, float], **extra) -> None:
    for metric, value in metrics.items():
        add_row(rows, model, metric_group, metric, value, **extra)


def mass_share_metrics(pred: np.ndarray, esa: np.ndarray, built_class: int) -> Dict[str, float]:
    total = float(np.sum(pred))
    built = esa == built_class
    nonbuilt = esa != built_class
    hard_nonbuilt = np.isin(esa, list(HARD_NONBUILT_CLASSES))

    mass_built = float(np.sum(pred[built]))
    mass_nonbuilt = float(np.sum(pred[nonbuilt]))
    mass_hard_nonbuilt = float(np.sum(pred[hard_nonbuilt]))

    metrics = {
        "valid_pixel_count": float(pred.size),
        "esa_built_pixel_count": float(np.sum(built)),
        "esa_built_prevalence": safe_div(float(np.sum(built)), float(pred.size)),
        "esa_nonbuilt_prevalence": safe_div(float(np.sum(nonbuilt)), float(pred.size)),
        "mass_total": total,
        "mass_in_esa_built": mass_built,
        "share_mass_in_esa_built": safe_div(mass_built, total),
        "mass_in_esa_nonbuilt": mass_nonbuilt,
        "share_mass_in_esa_nonbuilt": safe_div(mass_nonbuilt, total),
        "mass_in_hard_nonbuilt": mass_hard_nonbuilt,
        "share_mass_in_hard_nonbuilt": safe_div(mass_hard_nonbuilt, total),
    }

    for class_code, slug in INTERPRETABLE_LEAKAGE_CLASSES.items():
        class_mask = esa == class_code
        class_mass = float(np.sum(pred[class_mask]))
        metrics[f"mass_in_{slug}"] = class_mass
        metrics[f"share_mass_in_{slug}"] = safe_div(class_mass, total)

    return metrics


def hard_nonbuilt_by_class_rows(
    rows: List[dict],
    model: str,
    pred: np.ndarray,
    esa: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    total_pixels = float(pred.size)
    total_mass = float(np.sum(pred))
    out: Dict[str, Dict[str, float]] = {}

    for class_code, slug in INTERPRETABLE_LEAKAGE_CLASSES.items():
        mask = esa == class_code
        vals = pred[mask]
        mass = float(np.sum(vals))
        metrics = {
            "class_pixel_count": float(vals.size),
            "class_area_share": safe_div(float(vals.size), total_pixels),
            "pred_mass_sum": mass,
            "pred_mass_share": safe_div(mass, total_mass),
            "pred_mean": float(np.mean(vals)) if vals.size else np.nan,
            "pred_median": float(np.median(vals)) if vals.size else np.nan,
            "pred_p95": finite_percentile(vals, 95),
        }
        label = ESA_CLASS_LABELS.get(class_code, f"Class {class_code}")
        out[slug] = {
            "esa_class": class_code,
            "label": label,
            "metrics": metrics,
        }
        for metric, value in metrics.items():
            add_row(
                rows,
                model,
                "hard_nonbuilt_by_class",
                metric,
                value,
                esa_class=class_code,
                esa_class_label=label,
                group=slug,
            )
    return out


def class_summary_rows(rows: List[dict], model: str, pred: np.ndarray, esa: np.ndarray) -> Dict[str, Dict[str, float]]:
    total_pixels = float(pred.size)
    total_mass = float(np.sum(pred))
    out: Dict[str, Dict[str, float]] = {}

    for class_code in sorted(int(x) for x in np.unique(esa[np.isfinite(esa)])):
        mask = esa == class_code
        vals = pred[mask]
        metrics = {
            "class_pixel_count": float(vals.size),
            "class_area_share": safe_div(float(vals.size), total_pixels),
            "pred_mass_sum": float(np.sum(vals)),
            "pred_mass_share": safe_div(float(np.sum(vals)), total_mass),
            "pred_mean": float(np.mean(vals)) if vals.size else np.nan,
            "pred_median": float(np.median(vals)) if vals.size else np.nan,
            "pred_p95": finite_percentile(vals, 95),
        }
        label = ESA_CLASS_LABELS.get(class_code, f"Class {class_code}")
        out[str(class_code)] = {"label": label, "metrics": metrics}
        for metric, value in metrics.items():
            add_row(
                rows,
                model,
                "esa_class_summary",
                metric,
                value,
                esa_class=class_code,
                esa_class_label=label,
            )
    return out


def built_nonbuilt_contrast(pred: np.ndarray, esa: np.ndarray, built_class: int) -> Dict[str, float]:
    built_vals = pred[esa == built_class]
    nonbuilt_vals = pred[esa != built_class]

    mean_built = float(np.mean(built_vals)) if built_vals.size else np.nan
    mean_nonbuilt = float(np.mean(nonbuilt_vals)) if nonbuilt_vals.size else np.nan
    median_built = float(np.median(built_vals)) if built_vals.size else np.nan
    median_nonbuilt = float(np.median(nonbuilt_vals)) if nonbuilt_vals.size else np.nan

    return {
        "mean_pred_esa_built": mean_built,
        "mean_pred_esa_nonbuilt": mean_nonbuilt,
        "median_pred_esa_built": median_built,
        "median_pred_esa_nonbuilt": median_nonbuilt,
        "built_nonbuilt_mean_ratio": safe_div(mean_built, mean_nonbuilt),
        "built_nonbuilt_median_ratio": safe_div(median_built, median_nonbuilt),
    }


def topk_metrics(pred: np.ndarray, esa: np.ndarray, built_class: int, fractions: Iterable[float]) -> Dict[str, dict]:
    total_mass = float(np.sum(pred))
    built_prevalence = safe_div(float(np.sum(esa == built_class)), float(esa.size))
    out: Dict[str, dict] = {}

    for frac in fractions:
        top_mask, threshold = top_fraction_mask(pred, frac)
        top_pred = pred[top_mask]
        top_esa = esa[top_mask]
        top_mass = float(np.sum(top_pred))
        top_built = top_esa == built_class
        top_mass_built = float(np.sum(top_pred[top_built]))
        topk_esa_built_share = safe_div(float(np.sum(top_built)), float(np.sum(top_mask)))
        topk_mass_in_esa_built_share = safe_div(top_mass_built, top_mass)
        metrics = {
            "topk_frac": float(frac),
            "topk_pixel_count": float(np.sum(top_mask)),
            "topk_prediction_threshold": threshold,
            "esa_built_prevalence": built_prevalence,
            "topk_esa_built_share": topk_esa_built_share,
            "topk_pixel_esa_built_share": topk_esa_built_share,
            "topk_esa_built_lift": safe_div(topk_esa_built_share, built_prevalence),
            "topk_pixel_lift_over_esa_built_prevalence": safe_div(topk_esa_built_share, built_prevalence),
            "topk_mass_share": safe_div(top_mass, total_mass),
            "topk_mass_in_esa_built_share": topk_mass_in_esa_built_share,
            "topk_predicted_mass_in_esa_built_share": topk_mass_in_esa_built_share,
        }
        out[str(frac)] = metrics
    return out


def prevalence_matched_metrics(pred: np.ndarray, esa: np.ndarray, built_class: int) -> Dict[str, float]:
    esa_built = esa == built_class
    prevalence = safe_div(float(np.sum(esa_built)), float(esa_built.size))
    if not np.isfinite(prevalence) or prevalence <= 0:
        metrics = {
            "esa_built_prevalence": prevalence,
            "pred_binary_threshold_prevalence_matched": np.nan,
        }
        metrics.update({f"{k}_prevalence_matched": np.nan for k in [
            "precision", "recall", "f1", "iou", "balanced_accuracy"
        ]})
        return metrics

    pred_built, threshold = top_fraction_mask(pred, prevalence)
    agreement = binary_agreement(pred_built, esa_built)
    return {
        "esa_built_prevalence": prevalence,
        "pred_binary_threshold_prevalence_matched": threshold,
        "precision_prevalence_matched": agreement["precision"],
        "recall_prevalence_matched": agreement["recall"],
        "f1_prevalence_matched": agreement["f1"],
        "iou_prevalence_matched": agreement["iou"],
        "balanced_accuracy_prevalence_matched": agreement["balanced_accuracy"],
    }


def fixed_threshold_metrics(
    pred: np.ndarray,
    esa: np.ndarray,
    built_class: int,
    thresholds: Iterable[float],
) -> Dict[str, Dict[str, float]]:
    esa_built = esa == built_class
    out: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        agreement = binary_agreement(pred > threshold, esa_built)
        out[str(threshold)] = {
            "precision_fixed_threshold": agreement["precision"],
            "recall_fixed_threshold": agreement["recall"],
            "f1_fixed_threshold": agreement["f1"],
            "iou_fixed_threshold": agreement["iou"],
            "balanced_accuracy_fixed_threshold": agreement["balanced_accuracy"],
            "pred_built_prevalence_fixed_threshold": safe_div(float(np.sum(pred > threshold)), float(pred.size)),
        }
    return out


def wsf_conditioned_metrics(
    pred: np.ndarray,
    esa: np.ndarray,
    wsf: np.ndarray,
    built_class: int,
) -> Dict[str, Dict[str, float]]:
    groups = {
        "wsf_positive": wsf > 0,
        "wsf_negative": wsf <= 0,
        "esa_built_wsf_positive": (esa == built_class) & (wsf > 0),
        "esa_built_wsf_negative": (esa == built_class) & (wsf <= 0),
        "esa_nonbuilt_wsf_positive": (esa != built_class) & (wsf > 0),
        "esa_nonbuilt_wsf_negative": (esa != built_class) & (wsf <= 0),
    }
    total_pixels = float(pred.size)
    total_mass = float(np.sum(pred))
    out: Dict[str, Dict[str, float]] = {}

    for group, mask in groups.items():
        vals = pred[mask]
        mass = float(np.sum(vals))
        out[group] = {
            "pixel_count": float(vals.size),
            "pixel_share": safe_div(float(vals.size), total_pixels),
            "predicted_mass_sum": mass,
            "predicted_mass_share": safe_div(mass, total_mass),
            "mean_prediction": float(np.mean(vals)) if vals.size else np.nan,
            "median_prediction": float(np.median(vals)) if vals.size else np.nan,
        }
    return out


def add_model_deltas(rows: List[dict], nested: Dict[str, dict], models: List[str]) -> Dict[str, dict]:
    comparison_models = [m for m in ["wsf_uniform", "embed_wsf"] if m in nested]
    delta_metrics = [
        "mass_in_hard_nonbuilt",
        "share_mass_in_hard_nonbuilt",
        "mass_in_water",
        "share_mass_in_water",
        "mass_in_wetland",
        "share_mass_in_wetland",
        "mass_in_mangroves",
        "share_mass_in_mangroves",
        "mass_in_bare_sparse_vegetation",
        "share_mass_in_bare_sparse_vegetation",
        "mass_in_cropland",
        "share_mass_in_cropland",
        "mass_in_grassland",
        "share_mass_in_grassland",
    ]
    out: Dict[str, dict] = {}

    for model in models:
        out[model] = {}
        model_mass = nested.get(model, {}).get("mass_share", {})
        for comparison_model in comparison_models:
            comparison_mass = nested.get(comparison_model, {}).get("mass_share", {})
            out[model][comparison_model] = {}
            for metric in delta_metrics:
                value = model_mass.get(metric, np.nan)
                comparison_value = comparison_mass.get(metric, np.nan)
                delta = value - comparison_value if np.isfinite(value) and np.isfinite(comparison_value) else np.nan
                out[model][comparison_model][f"{metric}_delta"] = delta
                add_row(
                    rows,
                    model,
                    "model_delta",
                    f"{metric}_delta",
                    delta,
                    comparison_model=comparison_model,
                )
    return out


def write_leakage_maps(
    model: str,
    pred_full: np.ndarray,
    esa_full: np.ndarray,
    valid_mask: np.ndarray,
    template_profile: dict,
    out_dir: Path,
) -> Dict[str, str]:
    import rasterio

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model)
    hard_mask = valid_mask & np.isin(esa_full, list(HARD_NONBUILT_CLASSES))
    water_wetland_mask = valid_mask & np.isin(esa_full, [80, 90, 95])

    outputs = {
        "hard_nonbuilt_mass": (
            out_dir / f"{safe_model}_hard_nonbuilt_predicted_mass.tif",
            np.where(hard_mask, pred_full, 0.0),
            "float32",
            np.nan,
        ),
        "water_wetland_mangrove_mass": (
            out_dir / f"{safe_model}_water_wetland_mangrove_predicted_mass.tif",
            np.where(water_wetland_mask, pred_full, 0.0),
            "float32",
            np.nan,
        ),
    }

    written: Dict[str, str] = {}
    for label, (path, arr, dtype, nodata) in outputs.items():
        arr = arr.astype(dtype, copy=False)
        arr[~valid_mask] = nodata
        profile = template_profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            driver="GTiff",
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress="deflate",
            predictor=3 if dtype.startswith("float") else 2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        written[label] = str(path)
    return written


def write_metrics_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = [
        "model",
        "metric_group",
        "metric",
        "value",
        "esa_class",
        "esa_class_label",
        "topk_frac",
        "threshold",
        "group",
        "comparison_model",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def value_from_rows(rows: List[dict], model: str, metric_group: str, metric: str, **filters) -> Optional[float]:
    for row in rows:
        if row["model"] != model or row["metric_group"] != metric_group or row["metric"] != metric:
            continue
        ok = True
        for key, value in filters.items():
            if row.get(key) != value:
                ok = False
                break
        if ok:
            return row["value"]
    return None


def write_summary_figure(rows: List[dict], models: List[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    share_built = [
        value_from_rows(rows, m, "mass_share", "share_mass_in_esa_built") or 0.0
        for m in models
    ]
    axes[0, 0].bar(models, share_built)
    axes[0, 0].set_title("Predicted mass in ESA built-up")
    axes[0, 0].set_ylabel("Share of predicted mass")
    axes[0, 0].tick_params(axis="x", rotation=25)

    leakage_groups = [
        ("Hard", [(80, "Water"), (90, "Wetland"), (95, "Mangroves"), (70, "Snow/ice")]),
        ("Soft", [(30, "Grassland"), (40, "Cropland"), (60, "Bare/sparse")]),
    ]
    x_base = np.arange(len(models), dtype=float)
    bar_width = 0.34
    class_colors = {
        80: "#2b6cb0",
        90: "#38a169",
        95: "#276749",
        70: "#90cdf4",
        30: "#9ae6b4",
        40: "#ecc94b",
        60: "#c05621",
    }
    legend_seen = set()
    for group_idx, (group_label, classes) in enumerate(leakage_groups):
        x = x_base + (group_idx - 0.5) * bar_width
        bottom = np.zeros(len(models), dtype=float)
        for class_code, class_label in classes:
            vals = [
                value_from_rows(
                    rows,
                    model,
                    "hard_nonbuilt_by_class",
                    "pred_mass_sum",
                    esa_class=class_code,
                ) or 0.0
                for model in models
            ]
            label = class_label if class_code not in legend_seen else None
            axes[0, 1].bar(
                x,
                vals,
                width=bar_width,
                bottom=bottom,
                color=class_colors.get(class_code),
                label=label,
            )
            bottom += np.asarray(vals, dtype=float)
            legend_seen.add(class_code)
        for xpos, height in zip(x, bottom):
            axes[0, 1].text(
                xpos,
                height,
                group_label,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    axes[0, 1].set_xticks(x_base)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].set_title("Absolute predicted mass outside ESA built-up, by ESA class")
    axes[0, 1].set_ylabel("Predicted built-up mass")
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].legend(fontsize=7, ncol=2)

    top_rows = [
        r for r in rows
        if r["metric_group"] == "topk" and r["metric"] == "topk_esa_built_share"
    ]
    fracs = sorted({float(r["topk_frac"]) for r in top_rows if r["topk_frac"] is not None})
    x = np.arange(len(fracs))
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        pixel_vals = [
            value_from_rows(rows, model, "topk", "topk_esa_built_share", topk_frac=frac) or 0.0
            for frac in fracs
        ]
        mass_vals = [
            value_from_rows(rows, model, "topk", "topk_predicted_mass_in_esa_built_share", topk_frac=frac) or 0.0
            for frac in fracs
        ]
        axes[1, 0].bar(x + i * width, pixel_vals, width=width, alpha=0.55, label=f"{model} pixel")
        axes[1, 0].plot(x + i * width, mass_vals, marker="o", linewidth=1.5, label=f"{model} mass")
    axes[1, 0].set_title("Top-k ESA built-up overlap: pixels and predicted mass")
    axes[1, 0].set_ylabel("Built-up share")
    axes[1, 0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1, 0].set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    axes[1, 0].legend()

    class_rows = [
        r for r in rows
        if r["metric_group"] == "esa_class_summary" and r["metric"] == "pred_mass_share"
    ]
    class_codes = sorted({int(r["esa_class"]) for r in class_rows if r["esa_class"] is not None})
    bottom = np.zeros(len(models), dtype=float)
    for class_code in class_codes:
        vals = [
            value_from_rows(
                rows,
                model,
                "esa_class_summary",
                "pred_mass_share",
                esa_class=class_code,
            ) or 0.0
            for model in models
        ]
        label = ESA_CLASS_LABELS.get(class_code, str(class_code))
        axes[1, 1].bar(models, vals, bottom=bottom, label=label)
        bottom += np.asarray(vals, dtype=float)
    axes[1, 1].set_title("Predicted mass share by ESA class")
    axes[1, 1].set_ylabel("Share of predicted mass")
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    pred_paths = [Path(p).expanduser().resolve() for p in args.predictions]
    names = args.names if args.names is not None else [p.stem for p in pred_paths]
    if len(names) != len(pred_paths):
        raise ValueError("--names must have the same length as --predictions")

    if any(not 0 < f < 1 for f in args.topk_fracs):
        raise ValueError("--topk-fracs values must be in (0, 1)")

    template_arr, template_profile = read_single_band(pred_paths[0], categorical=False)
    predictions: Dict[str, np.ndarray] = {}
    for i, (name, path) in enumerate(zip(names, pred_paths)):
        arr, profile = read_single_band(path, categorical=False)
        if i > 0:
            require_same_grid(profile, template_profile, str(path), str(pred_paths[0]))
        predictions[name] = arr

    esa_path = Path(args.esa_worldcover).expanduser().resolve()
    esa, esa_profile = read_single_band(esa_path, categorical=True)
    require_same_grid(
        esa_profile,
        template_profile,
        str(esa_path),
        "prediction",
        esa_message=True,
    )

    wsf = None
    if args.wsf:
        wsf_path = Path(args.wsf).expanduser().resolve()
        wsf, wsf_profile = read_single_band(wsf_path, categorical=True)
        require_same_grid(wsf_profile, template_profile, str(wsf_path), str(pred_paths[0]))

    base_mask = np.isfinite(esa)
    if args.cell_ids:
        cell_ids_path = Path(args.cell_ids).expanduser().resolve()
        cell_ids, cell_profile = read_single_band(cell_ids_path, categorical=True)
        require_same_grid(cell_profile, template_profile, str(cell_ids_path), str(pred_paths[0]))
        base_mask &= np.isfinite(cell_ids) & (cell_ids > 0)

    rows: List[dict] = []
    nested: Dict[str, dict] = {}
    map_outputs: Dict[str, Dict[str, str]] = {}
    output_map_dir = Path(args.output_map_dir).expanduser().resolve() if args.output_map_dir else None

    for model, pred_full in predictions.items():
        print(f"[INFO] Evaluating {model}")
        valid = base_mask & np.isfinite(pred_full)
        if not np.any(valid):
            raise ValueError(f"No valid pixels for model: {model}")

        pred = pred_full[valid].astype(np.float64, copy=False)
        esa_valid = esa[valid].astype(np.int32, copy=False)

        nested[model] = {}

        mass_metrics = mass_share_metrics(pred, esa_valid, args.built_class)
        nested[model]["mass_share"] = mass_metrics
        add_metrics(rows, model, "mass_share", mass_metrics)

        class_metrics = class_summary_rows(rows, model, pred, esa_valid)
        nested[model]["esa_class_summary"] = class_metrics

        contrast = built_nonbuilt_contrast(pred, esa_valid, args.built_class)
        nested[model]["built_nonbuilt_contrast"] = contrast
        add_metrics(rows, model, "built_nonbuilt_contrast", contrast)

        topk = topk_metrics(pred, esa_valid, args.built_class, args.topk_fracs)
        nested[model]["topk"] = topk
        for frac_key, metrics in topk.items():
            frac = float(frac_key)
            add_metrics(rows, model, "topk", metrics, topk_frac=frac)

        hard_by_class = hard_nonbuilt_by_class_rows(rows, model, pred, esa_valid)
        nested[model]["hard_nonbuilt_by_class"] = hard_by_class

        prevalence = prevalence_matched_metrics(pred, esa_valid, args.built_class)
        nested[model]["prevalence_matched"] = prevalence
        add_metrics(rows, model, "prevalence_matched", prevalence)

        fixed = fixed_threshold_metrics(pred, esa_valid, args.built_class, args.min_pred_thresholds)
        nested[model]["fixed_threshold"] = fixed
        for threshold_key, metrics in fixed.items():
            add_metrics(rows, model, "fixed_threshold", metrics, threshold=float(threshold_key))

        if wsf is not None:
            wsf_valid_mask = valid & np.isfinite(wsf)
            if np.any(wsf_valid_mask):
                wsf_metrics = wsf_conditioned_metrics(
                    pred_full[wsf_valid_mask].astype(np.float64, copy=False),
                    esa[wsf_valid_mask].astype(np.int32, copy=False),
                    wsf[wsf_valid_mask].astype(np.float64, copy=False),
                    args.built_class,
                )
            else:
                wsf_metrics = {}
            nested[model]["wsf_conditioned"] = wsf_metrics
            for group, metrics in wsf_metrics.items():
                add_metrics(rows, model, "wsf_conditioned", metrics, group=group)

        if output_map_dir is not None:
            map_outputs[model] = write_leakage_maps(
                model=model,
                pred_full=pred_full,
                esa_full=esa,
                valid_mask=valid,
                template_profile=template_profile,
                out_dir=output_map_dir,
            )

    deltas = add_model_deltas(rows, nested, names)
    for model, model_deltas in deltas.items():
        nested.setdefault(model, {})["model_delta"] = model_deltas

    out_csv = Path(args.output_csv).expanduser().resolve()
    write_metrics_csv(rows, out_csv)
    print(f"[INFO] Saved metrics CSV to: {out_csv}")

    if args.output_json:
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": (
                "ESA WorldCover is categorical land cover. Metrics are categorical agreement "
                "and spatial plausibility diagnostics, not same-unit built-surface errors."
            ),
            "prediction_paths": {name: str(path) for name, path in zip(names, pred_paths)},
            "esa_worldcover_path": str(esa_path),
            "wsf_path": str(Path(args.wsf).expanduser().resolve()) if args.wsf else None,
            "cell_ids_path": str(Path(args.cell_ids).expanduser().resolve()) if args.cell_ids else None,
            "built_class": int(args.built_class),
            "topk_fracs": [float(x) for x in args.topk_fracs],
            "min_pred_thresholds": [float(x) for x in args.min_pred_thresholds],
            "hard_nonbuilt_classes": sorted(HARD_NONBUILT_CLASSES),
            "interpretable_leakage_classes": INTERPRETABLE_LEAKAGE_CLASSES,
            "esa_class_labels": ESA_CLASS_LABELS,
            "map_outputs": map_outputs,
            "metrics": nested,
        }
        out_json.write_text(json.dumps(json_ready(payload), indent=2))
        print(f"[INFO] Saved metrics JSON to: {out_json}")

    if args.output_fig:
        out_fig = Path(args.output_fig).expanduser().resolve()
        write_summary_figure(rows, names, out_fig)
        print(f"[INFO] Saved summary figure to: {out_fig}")

    if output_map_dir is not None:
        print(f"[INFO] Saved leakage maps to: {output_map_dir}")


if __name__ == "__main__":
    main()
