#!/usr/bin/env python3
"""
Quantitative evaluation of downscaled predictions against an external GHSL 10 m
reference product.

This script is intended to complement 08_compare_baselines.py. It treats the
fine GHSL layer as an external high-resolution reference/proxy, not as a direct
training label or exact ground truth. Metrics are reported at the native fine
grid and at coarser block-aggregation scales.

Example
-------
python 09_evaluate_against_ghsl10m.py \
  --predictions ~/data/outputs/wsf_uniform_baseline.tif \
                ~/data/outputs/embed_only_norm.tif \
                ~/data/outputs/embed_wsf_norm.tif \
  --names wsf_uniform embed_only embed_wsf \
  --reference ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --output-csv ~/data/outputs/evaluation_metrics.csv \
  --output-json ~/data/outputs/evaluation_metrics.json \
  --output-fig ~/data/outputs/evaluation_summary.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_FACTORS = [1, 5, 10, 25, 50, 100]
DEFAULT_TOP_FRACTIONS = [0.05, 0.10, 0.20]
DEFAULT_QUANTILES = [0.50, 0.75, 0.90, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate downscaled built-up predictions against an external GHSL 10 m reference."
    )
    p.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="Prediction rasters to evaluate. The first raster defines the target grid.",
    )
    p.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional model names, one per prediction raster.",
    )
    p.add_argument(
        "--reference",
        required=True,
        help="External fine-resolution GHSL reference raster, e.g. cropped_ghsl_raw_10m.tif.",
    )
    p.add_argument(
        "--wsf",
        default=None,
        help="Optional WSF binary/support raster for inside/outside diagnostics.",
    )
    p.add_argument(
        "--cell-ids",
        default=None,
        help="Optional fine-grid coarse-cell ID raster used only to restrict valid pixels to represented cells.",
    )
    p.add_argument("--output-csv", required=True, help="Long-format output CSV with metrics.")
    p.add_argument("--output-json", default=None, help="Optional JSON output with metrics and run metadata.")
    p.add_argument("--output-fig", default=None, help="Optional PNG summary figure.")
    p.add_argument(
        "--factors",
        nargs="+",
        type=int,
        default=DEFAULT_FACTORS,
        help="Block aggregation factors. Defaults to 1 5 10 25 50 100.",
    )
    p.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=DEFAULT_TOP_FRACTIONS,
        help="Top-reference fractions for overlap/concentration metrics.",
    )
    p.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=DEFAULT_QUANTILES,
        help="Distribution quantiles to compare.",
    )
    p.add_argument(
        "--built-threshold",
        type=float,
        default=0.0,
        help="Threshold for built/non-built support metrics. Defaults to >0.",
    )
    p.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.5,
        help="Minimum valid-pixel fraction for an aggregated block to be kept.",
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


def align_to_target(src_path: Path, target_profile: dict, resampling: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    resampling_enum = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
    }[resampling]

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
            resampling=resampling_enum,
        )
    return dst


def load_prediction(path: Path, target_profile: dict, is_template: bool) -> np.ndarray:
    arr, profile = read_single_band(path)
    if is_template or same_grid(profile, target_profile):
        return arr
    print(f"[WARN] Prediction grid differs from template; reprojecting with bilinear: {path}")
    return align_to_target(path, target_profile, "bilinear")


def block_sum(arr: np.ndarray, factor: int, min_valid_fraction: float) -> np.ndarray:
    if factor == 1:
        return arr.astype(np.float32, copy=False)
    if factor < 1:
        raise ValueError("Aggregation factors must be >= 1")

    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    if h2 == 0 or w2 == 0:
        raise ValueError(f"Aggregation factor {factor} is too large for raster shape {arr.shape}")

    cropped = arr[:h2, :w2]
    reshaped = cropped.reshape(h2 // factor, factor, w2 // factor, factor)
    valid = np.isfinite(reshaped)
    sums = np.where(valid, reshaped, 0.0).sum(axis=(1, 3))
    counts = valid.sum(axis=(1, 3))
    keep = counts >= (factor * factor * min_valid_fraction)
    out = sums.astype(np.float32, copy=False)
    out[~keep] = np.nan
    return out


def pixel_scale_m(profile: dict, factor: int) -> Optional[float]:
    transform = profile["transform"]
    crs = profile.get("crs")
    if crs is not None and getattr(crs, "is_projected", False):
        return float(abs(transform.a) * factor)
    return None


def valid_vectors(pred: np.ndarray, ref: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        valid &= mask
    return pred[valid].astype(np.float64, copy=False), ref[valid].astype(np.float64, copy=False)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    xr = average_ranks(x)
    yr = average_ranks(y)
    return pearson(xr, yr)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]

    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Ranks are 1-based; tied values receive the average rank.
        avg_rank = 0.5 * (start + 1 + stop)
        ranks[order[start:stop]] = avg_rank
        start = stop
    return ranks


def top_fraction_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("Top fractions must be in (0, 1)")
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool)
    n_top = max(1, int(np.ceil(values.size * fraction)))
    cutoff = np.partition(values, values.size - n_top)[values.size - n_top]
    return values >= cutoff


def support_metrics(pred: np.ndarray, ref: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_built = pred > threshold
    ref_built = ref > threshold
    tp = float(np.sum(pred_built & ref_built))
    fp = float(np.sum(pred_built & ~ref_built))
    fn = float(np.sum(~pred_built & ref_built))
    tn = float(np.sum(~pred_built & ~ref_built))

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else np.nan

    return {
        "built_precision": precision,
        "built_recall": recall,
        "built_f1": f1,
        "built_accuracy": accuracy,
    }


def scalar_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    built_threshold: float,
    top_fractions: Iterable[float],
    quantiles: Iterable[float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = int(pred.size)
    if n == 0:
        return {"n_pixels": 0}

    err = pred - ref
    pred_sum = float(np.sum(pred))
    ref_sum = float(np.sum(ref))
    abs_err = np.abs(err)

    out.update(
        {
            "n_pixels": float(n),
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "mean_pred": float(np.mean(pred)),
            "mean_ref": float(np.mean(ref)),
            "sum_pred": pred_sum,
            "sum_ref": ref_sum,
            "mass_ratio": pred_sum / ref_sum if ref_sum != 0 else np.nan,
            "pearson": pearson(pred, ref),
            "spearman": spearman(pred, ref),
        }
    )
    out.update(support_metrics(pred, ref, built_threshold))

    for q in quantiles:
        pred_q = float(np.quantile(pred, q))
        ref_q = float(np.quantile(ref, q))
        suffix = f"q{int(round(q * 100)):02d}"
        out[f"pred_{suffix}"] = pred_q
        out[f"ref_{suffix}"] = ref_q
        out[f"quantile_error_{suffix}"] = pred_q - ref_q

    for frac in top_fractions:
        pred_top = top_fraction_mask(pred, frac)
        ref_top = top_fraction_mask(ref, frac)
        overlap = float(np.mean(pred_top & ref_top) / frac)
        ref_top_mass_pred = float(np.sum(pred[ref_top]))
        pred_mass_top_ref = ref_top_mass_pred / pred_sum if pred_sum > 0 else np.nan
        ref_mass_top_ref = float(np.sum(ref[ref_top]) / ref_sum) if ref_sum > 0 else np.nan
        suffix = f"top{int(round(frac * 100)):02d}"
        out[f"{suffix}_overlap_ratio"] = overlap
        out[f"pred_mass_in_ref_{suffix}"] = pred_mass_top_ref
        out[f"ref_mass_in_ref_{suffix}"] = ref_mass_top_ref

    return out


def add_metric(
    rows: List[dict],
    model: str,
    scale_factor: int,
    scale_m: Optional[float],
    metric_group: str,
    metric_values: Dict[str, float],
) -> None:
    for metric, value in metric_values.items():
        rows.append(
            {
                "model": model,
                "scale_factor": int(scale_factor),
                "scale_m": scale_m,
                "metric_group": metric_group,
                "metric": metric,
                "value": value,
            }
        )


def wsf_diagnostics(
    pred: np.ndarray,
    ref: np.ndarray,
    wsf: np.ndarray,
    built_threshold: float,
    base_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    valid = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(wsf)
    if base_mask is not None:
        valid &= base_mask
    if not np.any(valid):
        return {"wsf_valid_pixels": 0}

    p = pred[valid].astype(np.float64, copy=False)
    r = ref[valid].astype(np.float64, copy=False)
    w = wsf[valid] > 0

    pred_sum = float(np.sum(p))
    ref_sum = float(np.sum(r))
    pred_inside = float(np.sum(p[w]))
    ref_inside = float(np.sum(r[w]))
    pred_outside = float(np.sum(p[~w]))
    ref_outside = float(np.sum(r[~w]))

    ref_built = r > built_threshold
    outside_ref_built = ref_built & ~w
    recovered_outside = (p > built_threshold) & outside_ref_built

    out = {
        "wsf_valid_pixels": float(p.size),
        "pred_mass_fraction_inside_wsf": pred_inside / pred_sum if pred_sum > 0 else np.nan,
        "ref_mass_fraction_inside_wsf": ref_inside / ref_sum if ref_sum > 0 else np.nan,
        "pred_mass_outside_wsf": pred_outside,
        "ref_mass_outside_wsf": ref_outside,
        "ref_built_pixels_outside_wsf": float(np.sum(outside_ref_built)),
        "pred_recall_ref_built_outside_wsf": (
            float(np.sum(recovered_outside)) / float(np.sum(outside_ref_built))
            if np.sum(outside_ref_built) > 0
            else np.nan
        ),
    }

    for label, mask in [("inside_wsf", w), ("outside_wsf", ~w)]:
        if np.sum(mask) < 2:
            out[f"mae_{label}"] = np.nan
            out[f"spearman_{label}"] = np.nan
            continue
        out[f"mae_{label}"] = float(np.mean(np.abs(p[mask] - r[mask])))
        out[f"spearman_{label}"] = spearman(p[mask], r[mask])
    return out


def write_metrics_csv(rows: List[dict], out_path: Path) -> None:
    fieldnames = ["model", "scale_factor", "scale_m", "metric_group", "metric", "value"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_figure(rows: List[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    def metric_rows(metric: str) -> List[dict]:
        sub = [r for r in rows if r["metric_group"] == "scale" and r["metric"] == metric]
        return sorted(sub, key=lambda r: (r["model"], r["scale_factor"]))

    def group_by_model(metric_data: List[dict]) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for row in metric_data:
            out.setdefault(row["model"], []).append(row)
        return out

    spearman_rows = metric_rows("spearman")
    rmse_rows = metric_rows("rmse")
    top_rows = [
        r for r in rows
        if r["metric_group"] == "scale" and r["scale_factor"] == 1 and r["metric"] == "top10_overlap_ratio"
    ]
    wsf_rows = [
        r for r in rows
        if r["metric_group"] == "wsf" and r["metric"] == "pred_mass_fraction_inside_wsf"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    for model, g in group_by_model(spearman_rows).items():
        x = [float(row["scale_m"] if row["scale_m"] is not None else row["scale_factor"]) for row in g]
        y = [float(row["value"]) for row in g]
        axes[0, 0].plot(x, y, marker="o", label=model)
    axes[0, 0].set_title("Rank agreement by aggregation scale")
    axes[0, 0].set_xlabel("Scale")
    axes[0, 0].set_ylabel("Spearman correlation")
    axes[0, 0].legend()

    for model, g in group_by_model(rmse_rows).items():
        x = [float(row["scale_m"] if row["scale_m"] is not None else row["scale_factor"]) for row in g]
        y = [float(row["value"]) for row in g]
        axes[0, 1].plot(x, y, marker="o", label=model)
    axes[0, 1].set_title("Error by aggregation scale")
    axes[0, 1].set_xlabel("Scale")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].legend()

    axes[1, 0].bar([r["model"] for r in top_rows], [r["value"] for r in top_rows])
    axes[1, 0].set_title("Native-grid top-10% overlap")
    axes[1, 0].set_ylabel("Overlap ratio")
    axes[1, 0].tick_params(axis="x", rotation=25)

    if wsf_rows:
        axes[1, 1].bar([r["model"] for r in wsf_rows], [r["value"] for r in wsf_rows])
        axes[1, 1].set_title("Predicted mass inside WSF")
        axes[1, 1].set_ylabel("Fraction")
        axes[1, 1].tick_params(axis="x", rotation=25)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "No WSF diagnostics requested", ha="center", va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    pred_paths = [Path(p).expanduser().resolve() for p in args.predictions]
    names = args.names if args.names is not None else [p.stem for p in pred_paths]
    if len(names) != len(pred_paths):
        raise ValueError("--names must have the same length as --predictions")

    if any(f < 1 for f in args.factors):
        raise ValueError("--factors must all be >= 1")
    if not 0 <= args.min_valid_fraction <= 1:
        raise ValueError("--min-valid-fraction must be in [0, 1]")

    template_arr, template_profile = read_single_band(pred_paths[0])
    predictions = {
        name: load_prediction(path, template_profile, is_template=(i == 0))
        for i, (name, path) in enumerate(zip(names, pred_paths))
    }

    reference_path = Path(args.reference).expanduser().resolve()
    reference = align_to_target(reference_path, template_profile, "bilinear")

    wsf = None
    if args.wsf:
        wsf = align_to_target(Path(args.wsf).expanduser().resolve(), template_profile, "nearest")

    base_mask = np.isfinite(reference)
    if args.cell_ids:
        cell_ids = align_to_target(Path(args.cell_ids).expanduser().resolve(), template_profile, "nearest")
        base_mask &= np.isfinite(cell_ids) & (cell_ids > 0)

    rows: List[dict] = []
    nested: Dict[str, dict] = {}

    for name, pred in predictions.items():
        nested[name] = {"scale": {}, "wsf": None}
        print(f"[INFO] Evaluating {name}")
        pred_base = np.where(base_mask, pred, np.nan)
        reference_base = np.where(base_mask, reference, np.nan)

        for factor in args.factors:
            pred_agg = block_sum(pred_base, factor, args.min_valid_fraction)
            ref_agg = block_sum(reference_base, factor, args.min_valid_fraction)

            p, r = valid_vectors(pred_agg, ref_agg)
            metrics = scalar_metrics(
                pred=p,
                ref=r,
                built_threshold=args.built_threshold,
                top_fractions=args.top_fractions,
                quantiles=args.quantiles,
            )
            scale_m = pixel_scale_m(template_profile, factor)
            nested[name]["scale"][str(factor)] = {
                "scale_m": scale_m,
                "metrics": metrics,
            }
            add_metric(rows, name, factor, scale_m, "scale", metrics)

        if wsf is not None:
            metrics = wsf_diagnostics(pred, reference, wsf, args.built_threshold, base_mask)
            nested[name]["wsf"] = metrics
            add_metric(rows, name, 1, pixel_scale_m(template_profile, 1), "wsf", metrics)

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(rows, out_csv)
    print(f"[INFO] Saved metrics CSV to: {out_csv}")

    if args.output_json:
        out_json = Path(args.output_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "note": (
                "Reference is treated as an external high-resolution proxy, not direct ground truth "
                "or a training label."
            ),
            "prediction_paths": {name: str(path) for name, path in zip(names, pred_paths)},
            "reference_path": str(reference_path),
            "wsf_path": str(Path(args.wsf).expanduser().resolve()) if args.wsf else None,
            "cell_ids_path": str(Path(args.cell_ids).expanduser().resolve()) if args.cell_ids else None,
            "factors": [int(f) for f in args.factors],
            "top_fractions": [float(f) for f in args.top_fractions],
            "quantiles": [float(q) for q in args.quantiles],
            "built_threshold": float(args.built_threshold),
            "min_valid_fraction": float(args.min_valid_fraction),
            "metrics": nested,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[INFO] Saved metrics JSON to: {out_json}")

    if args.output_fig:
        out_fig = Path(args.output_fig).expanduser().resolve()
        write_summary_figure(rows, out_fig)
        print(f"[INFO] Saved summary figure to: {out_fig}")


if __name__ == "__main__":
    main()
