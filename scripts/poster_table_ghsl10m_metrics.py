#!/usr/bin/env python3
"""
poster_table_ghsl10m_metrics.py

Create a compact poster table comparing three downscaling outputs against the
GHSL 2018 10 m pseudo-reference.

Metrics
-------
- RMSE
- MAE
- Pearson correlation
- SSIM
- Leakage into GHSL-10 m zero pixels (% of predicted mass)
- Allocation outside WSF support (% of predicted mass)

Example
-------
mamba run -n diss python scripts/poster_table_ghsl10m_metrics.py \
  --output-dir ~/data/outputs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


DEFAULT_MODELS = [
    ("WSF-only", "~/data/outputs/wsf_uniform_baseline.tif"),
    ("Embeddings-only", "~/data/outputs/embed_only_norm.tif"),
    ("Prior-corrected WSF + embeddings", "~/data/outputs/prior_corrected_wsf_diffnorm_norm.tif"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create compact poster metrics table against GHSL 10 m pseudo-reference.")
    p.add_argument(
        "--predictions",
        nargs="+",
        default=[path for _, path in DEFAULT_MODELS],
        help="Prediction rasters. Defaults to WSF-only, embeddings-only, prior-corrected WSF+embeddings.",
    )
    p.add_argument(
        "--names",
        nargs="+",
        default=[name for name, _ in DEFAULT_MODELS],
        help="Model names, one per prediction raster.",
    )
    p.add_argument(
        "--reference",
        default="~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif",
        help="GHSL 2018 10 m pseudo-reference raster.",
    )
    p.add_argument("--wsf", default="~/data/WSF_Data/cropped_wsf.tif", help="WSF binary support raster.")
    p.add_argument(
        "--cell-ids",
        default="~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif",
        help="Optional fine-grid coarse-cell IDs used to restrict valid pixels.",
    )
    p.add_argument("--zero-threshold", type=float, default=0.0, help="Reference values <= this are treated as zero.")
    p.add_argument("--wsf-threshold", type=float, default=0.0, help="WSF values > this are treated as support.")
    p.add_argument("--ssim-window", type=int, default=11, help="Odd local-window size for masked SSIM.")
    p.add_argument(
        "--ssim-min-valid-fraction",
        type=float,
        default=0.8,
        help="Minimum valid-pixel fraction for a local SSIM window.",
    )
    p.add_argument("--output-dir", default="~/data/outputs", help="Directory for table outputs.")
    p.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    p.add_argument("--output-json", default=None, help="Optional JSON output path.")
    p.add_argument("--output-md", default=None, help="Optional Markdown table output path.")
    p.add_argument("--output-tex", default=None, help="Optional LaTeX table output path.")
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

    resampling_enum = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear}[resampling]
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


def load_aligned(path: Path, target_profile: dict, resampling: str) -> np.ndarray:
    arr, profile = read_single_band(path)
    if same_grid(profile, target_profile):
        return arr
    print(f"[WARN] Reprojecting to reference grid with {resampling}: {path}")
    return align_to_target(path, target_profile, resampling)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def masked_local_ssim(
    pred: np.ndarray,
    ref: np.ndarray,
    valid_mask: np.ndarray,
    window_size: int,
    min_valid_fraction: float,
) -> float:
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("--ssim-window must be an odd integer >= 3")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("--ssim-min-valid-fraction must be in (0, 1]")

    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return global_ssim(pred[valid_mask], ref[valid_mask])

    valid = valid_mask & np.isfinite(pred) & np.isfinite(ref)
    if np.sum(valid) < 2:
        return np.nan

    p = np.where(valid, pred, 0).astype(np.float64, copy=False)
    r = np.where(valid, ref, 0).astype(np.float64, copy=False)
    w = valid.astype(np.float64)

    local_weight = uniform_filter(w, size=window_size, mode="constant", cval=0.0)
    keep = local_weight >= min_valid_fraction
    if not np.any(keep):
        return np.nan

    sum_p = uniform_filter(p * w, size=window_size, mode="constant", cval=0.0)
    sum_r = uniform_filter(r * w, size=window_size, mode="constant", cval=0.0)
    sum_p2 = uniform_filter((p**2) * w, size=window_size, mode="constant", cval=0.0)
    sum_r2 = uniform_filter((r**2) * w, size=window_size, mode="constant", cval=0.0)
    sum_pr = uniform_filter((p * r) * w, size=window_size, mode="constant", cval=0.0)

    mux = np.divide(sum_p, local_weight, out=np.zeros_like(sum_p), where=local_weight > 0)
    muy = np.divide(sum_r, local_weight, out=np.zeros_like(sum_r), where=local_weight > 0)
    varx = np.divide(sum_p2, local_weight, out=np.zeros_like(sum_p2), where=local_weight > 0) - mux**2
    vary = np.divide(sum_r2, local_weight, out=np.zeros_like(sum_r2), where=local_weight > 0) - muy**2
    covxy = np.divide(sum_pr, local_weight, out=np.zeros_like(sum_pr), where=local_weight > 0) - mux * muy

    p_valid = pred[valid].astype(np.float64, copy=False)
    r_valid = ref[valid].astype(np.float64, copy=False)
    data_range = float(max(np.max(p_valid), np.max(r_valid)) - min(np.min(p_valid), np.min(r_valid)))
    if not np.isfinite(data_range) or data_range <= 0:
        return np.nan

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mux * muy + c1) * (2 * covxy + c2)) / ((mux**2 + muy**2 + c1) * (varx + vary + c2))
    finite_keep = keep & np.isfinite(ssim_map)
    return float(np.mean(ssim_map[finite_keep])) if np.any(finite_keep) else np.nan


def global_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Fallback global SSIM over valid vectors."""
    if x.size < 2:
        return np.nan

    data_range = float(max(np.max(x), np.max(y)) - min(np.min(x), np.min(y)))
    if not np.isfinite(data_range) or data_range <= 0:
        return np.nan

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    covxy = float(np.mean((x - mux) * (y - muy)))

    numerator = (2 * mux * muy + c1) * (2 * covxy + c2)
    denominator = (mux**2 + muy**2 + c1) * (varx + vary + c2)
    return numerator / denominator if denominator != 0 else np.nan


def compute_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    wsf: np.ndarray,
    valid_mask: np.ndarray,
    zero_threshold: float,
    wsf_threshold: float,
    ssim_window: int,
    ssim_min_valid_fraction: float,
) -> Dict[str, float]:
    valid = valid_mask & np.isfinite(pred) & np.isfinite(ref) & np.isfinite(wsf)
    p = pred[valid].astype(np.float64, copy=False)
    r = ref[valid].astype(np.float64, copy=False)
    w = wsf[valid].astype(np.float64, copy=False)

    if p.size == 0:
        raise ValueError("No valid pixels available for metric computation.")

    err = p - r
    pred_sum = float(np.sum(p))
    zero_ref = r <= zero_threshold
    outside_wsf = w <= wsf_threshold

    pred_mass_zero_ref = float(np.sum(p[zero_ref]))
    pred_mass_outside_wsf = float(np.sum(p[outside_wsf]))

    return {
        "n_pixels": float(p.size),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": pearson(p, r),
        "ssim": masked_local_ssim(pred, ref, valid, ssim_window, ssim_min_valid_fraction),
        "pred_sum": pred_sum,
        "reference_sum": float(np.sum(r)),
        "leakage_into_ghsl10m_zero_pixels_pct": 100.0 * pred_mass_zero_ref / pred_sum if pred_sum > 0 else np.nan,
        "allocation_outside_wsf_support_pct": 100.0 * pred_mass_outside_wsf / pred_sum if pred_sum > 0 else np.nan,
        "pred_mass_in_ghsl10m_zero_pixels": pred_mass_zero_ref,
        "pred_mass_outside_wsf_support": pred_mass_outside_wsf,
    }


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    csv_path = Path(args.output_csv).expanduser().resolve() if args.output_csv else output_dir / "poster_table_ghsl10m_metrics.csv"
    json_path = Path(args.output_json).expanduser().resolve() if args.output_json else output_dir / "poster_table_ghsl10m_metrics.json"
    md_path = Path(args.output_md).expanduser().resolve() if args.output_md else output_dir / "poster_table_ghsl10m_metrics.md"
    tex_path = Path(args.output_tex).expanduser().resolve() if args.output_tex else output_dir / "poster_table_ghsl10m_metrics.tex"
    return csv_path, json_path, md_path, tex_path


def format_value(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if metric in {"rmse", "mae"}:
        return f"{value:.3f}"
    if metric in {"pearson", "ssim"}:
        return f"{value:.3f}"
    if metric.endswith("_pct"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def display_rows(rows: list[dict]) -> list[dict]:
    names = {
        "rmse": "RMSE",
        "mae": "MAE",
        "pearson": "Pearson r",
        "ssim": "SSIM",
        "leakage_into_ghsl10m_zero_pixels_pct": "Leakage into GHSL-10 m zero pixels (%)",
        "allocation_outside_wsf_support_pct": "Allocation outside WSF support (%)",
    }
    out = []
    for row in rows:
        display = {"Model": row["model"]}
        for metric, label in names.items():
            display[label] = format_value(metric, row[metric])
        out.append(display)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "rmse",
        "mae",
        "pearson",
        "ssim",
        "leakage_into_ghsl10m_zero_pixels_pct",
        "allocation_outside_wsf_support_pct",
        "n_pixels",
        "pred_sum",
        "reference_sum",
        "pred_mass_in_ghsl10m_zero_pixels",
        "pred_mass_outside_wsf_support",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict], path: Path) -> None:
    display = display_rows(rows)
    headers = list(display[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_latex(rows: list[dict], path: Path) -> None:
    display = display_rows(rows)
    headers = list(display[0].keys())
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for row in display:
        lines.append(" & ".join(row[h] for h in headers) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if len(args.predictions) != len(args.names):
        raise ValueError("--predictions and --names must have the same length")

    reference_path = Path(args.reference).expanduser().resolve()
    wsf_path = Path(args.wsf).expanduser().resolve()
    cell_ids_path = Path(args.cell_ids).expanduser().resolve() if args.cell_ids else None
    csv_path, json_path, md_path, tex_path = resolve_outputs(args)

    reference, reference_profile = read_single_band(reference_path)
    wsf = load_aligned(wsf_path, reference_profile, "nearest")
    valid_mask = np.isfinite(reference)
    if cell_ids_path is not None and cell_ids_path.exists():
        cell_ids = load_aligned(cell_ids_path, reference_profile, "nearest")
        valid_mask &= np.isfinite(cell_ids) & (cell_ids > 0)

    rows = []
    prediction_paths = [Path(p).expanduser().resolve() for p in args.predictions]
    for name, path in zip(args.names, prediction_paths):
        pred = load_aligned(path, reference_profile, "bilinear")
        metrics = compute_metrics(
            pred=pred,
            ref=reference,
            wsf=wsf,
            valid_mask=valid_mask,
            zero_threshold=args.zero_threshold,
            wsf_threshold=args.wsf_threshold,
            ssim_window=args.ssim_window,
            ssim_min_valid_fraction=args.ssim_min_valid_fraction,
        )
        rows.append({"model": name, **metrics})
        print(
            f"[INFO] {name}: RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
            f"Pearson={metrics['pearson']:.3f}, SSIM={metrics['ssim']:.3f}"
        )

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_latex(rows, tex_path)

    payload = {
        "note": (
            "Predictions are compared to the GHSL 2018 10 m pseudo-reference. "
            "Leakage percentages are predicted-mass shares."
        ),
        "reference": str(reference_path),
        "wsf": str(wsf_path),
        "cell_ids": str(cell_ids_path) if cell_ids_path is not None else None,
        "zero_threshold": float(args.zero_threshold),
        "wsf_threshold": float(args.wsf_threshold),
        "ssim_window": int(args.ssim_window),
        "ssim_min_valid_fraction": float(args.ssim_min_valid_fraction),
        "predictions": {name: str(path) for name, path in zip(args.names, prediction_paths)},
        "metrics": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"[INFO] Saved CSV to: {csv_path}")
    print(f"[INFO] Saved Markdown table to: {md_path}")
    print(f"[INFO] Saved LaTeX table to: {tex_path}")
    print(f"[INFO] Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()
