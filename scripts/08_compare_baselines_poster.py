#!/usr/bin/env python3
"""
08_compare_baselines_poster.py

Create the poster version of Figure 2 using aggregated 10x10 maps only.

Panels
------
A. WSF-only allocation
B. Embeddings-only
C. Prior-corrected WSF + embeddings
D. GHSL 2018 10 m pseudo-reference

Example
-------
python scripts/08_compare_baselines_poster.py \
  --baseline0 ~/data/outputs/wsf_uniform_baseline.tif \
  --baseline1 ~/data/outputs/embed_only_norm.tif \
  --baseline3 ~/data/outputs/prior_corrected_wsf_diffnorm_norm.tif \
  --ghsl10m ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --output-dir ~/data/outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot poster comparison with aggregated baseline maps.")
    p.add_argument("--baseline0", required=True, help="Path to WSF-only allocation raster")
    p.add_argument("--baseline1", required=True, help="Path to embeddings-only raster")
    p.add_argument("--baseline3", required=True, help="Path to prior-corrected WSF + embeddings raster")
    p.add_argument("--ghsl10m", required=True, help="Path to GHSL 2018/2019 10 m pseudo-reference raster")
    p.add_argument("--agg-factor", type=int, default=10, help="Aggregation factor for displayed maps")
    p.add_argument("--vmax-percentile", type=float, default=98.0, help="Shared vmax percentile")
    p.add_argument("--output-dir", default="~/data/outputs", help="Directory for poster figure exports")
    p.add_argument(
        "--output",
        default=None,
        help="Optional PNG output path. Defaults to figure2_baseline_comparison_poster.png in --output-dir.",
    )
    p.add_argument(
        "--svg-output",
        default=None,
        help="Optional SVG output path. Defaults to figure2_baseline_comparison_poster.svg in --output-dir.",
    )
    p.add_argument(
        "--pdf-output",
        default=None,
        help="Optional PDF output path. If omitted, no PDF is exported.",
    )
    return p.parse_args()


def read_single_band(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
    nodata = profile.get("nodata", None)
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def ensure_same_grid(a: dict, b: dict, name_a: str, name_b: str) -> None:
    for k in ["height", "width", "crs", "transform"]:
        if a[k] != b[k]:
            raise ValueError(f"{name_a} and {name_b} differ on {k}: {a[k]} vs {b[k]}")


def align_to_target(src_path: Path, target_profile: dict, resampling: Resampling) -> np.ndarray:
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


def aggregate_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    cropped = arr[:h2, :w2]

    reshaped = cropped.reshape(h2 // factor, factor, w2 // factor, factor)
    valid = np.isfinite(reshaped)
    sums = np.where(valid, reshaped, 0.0).sum(axis=(1, 3))
    counts = valid.sum(axis=(1, 3))
    out = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=np.float32), where=counts > 0)
    return out.astype(np.float32, copy=False)


def aggregated_extent(profile: dict, factor: int) -> Tuple[float, float, float, float]:
    transform = profile["transform"]
    width = profile["width"] // factor
    height = profile["height"] // factor
    left = transform.c
    top = transform.f
    right = left + transform.a * factor * width
    bottom = top + transform.e * factor * height
    return (left, right, bottom, top)


def percentile_vmax(arrays, percentile: float) -> float:
    vals = []
    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 1.0

    vmax = float(np.percentile(np.concatenate(vals), percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.concatenate(vals)))
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    png_path = Path(args.output).expanduser().resolve() if args.output else output_dir / "figure2_baseline_comparison_poster.png"
    svg_path = (
        Path(args.svg_output).expanduser().resolve()
        if args.svg_output
        else output_dir / "figure2_baseline_comparison_poster.svg"
    )
    pdf_path = Path(args.pdf_output).expanduser().resolve() if args.pdf_output else None
    return png_path, svg_path, pdf_path


def main():
    args = parse_args()

    agg_factor = int(args.agg_factor)
    if agg_factor < 1:
        raise ValueError("--agg-factor must be >= 1")

    b0_path = Path(args.baseline0).expanduser().resolve()
    b1_path = Path(args.baseline1).expanduser().resolve()
    b3_path = Path(args.baseline3).expanduser().resolve()
    ghsl10m_path = Path(args.ghsl10m).expanduser().resolve()
    png_path, svg_path, pdf_path = resolve_outputs(args)

    b0, p0 = read_single_band(b0_path)
    b1, p1 = read_single_band(b1_path)
    b3, p3 = read_single_band(b3_path)

    ensure_same_grid(p0, p1, "baseline0", "baseline1")
    ensure_same_grid(p0, p3, "baseline0", "baseline3")

    ghsl10 = align_to_target(ghsl10m_path, p0, Resampling.bilinear)

    arrays = [
        aggregate_mean(b0, agg_factor),
        aggregate_mean(b1, agg_factor),
        aggregate_mean(b3, agg_factor),
        aggregate_mean(ghsl10, agg_factor),
    ]
    titles = [
        "A. WSF-only allocation",
        "B. Embeddings-only",
        "C. Prior-corrected WSF + embeddings",
        "D. GHSL 2018 10 m pseudo-reference",
    ]

    vmin = 0.0
    vmax = percentile_vmax(arrays, args.vmax_percentile)
    extent = aggregated_extent(p0, agg_factor)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8), constrained_layout=True)
    image = None
    for ax, arr, title in zip(axes, arrays, titles):
        image = ax.imshow(arr, extent=extent, origin="upper", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)

    cbar = fig.colorbar(image, ax=axes, orientation="vertical", fraction=0.025, pad=0.012)
    cbar.set_label("m² per 10 m pixel", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    for path in [png_path, svg_path, pdf_path]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Shared scale: vmin={vmin:.3g}, vmax={vmax:.3g} ({args.vmax_percentile:g}th percentile)")
    print(f"[INFO] Saved PNG to: {png_path}")
    print(f"[INFO] Saved SVG to: {svg_path}")
    if pdf_path is not None:
        print(f"[INFO] Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
