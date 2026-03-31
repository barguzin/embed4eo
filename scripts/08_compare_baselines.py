#!/usr/bin/env python3
"""
08_compare_baselines.py

Create a comparison figure for three downscaling baselines plus a 10 m GHSL reference:
- Baseline 0: WSF-uniform
- Baseline 1: embeddings-only
- Baseline 2: embeddings + WSF-guided
- Reference: GHSL 2019 10 m

Main figure:
    2 rows x 4 columns
    top row    = fine-grid predictions / reference
    bottom row = same layers aggregated by factor N (default 10)

Optional context figure:
    1 row x 2 columns
    = GHSL coarse visualized on fine grid + WSF binary mask

Example
-------
python 08_compare_baselines.py \
  --baseline0 ~/data/outputs/wsf_uniform_baseline.tif \
  --baseline1 ~/data/outputs/embed_only_norm.tif \
  --baseline2 ~/data/outputs/embed_wsf_norm.tif \
  --ghsl10m ~/data/GHSL_BUILD/cropped_ghsl_raw_10m.tif \
  --output ~/data/outputs/baseline_comparison.png \
  --agg-factor 10 \
  --title "Baseline comparison with GHSL 10 m reference" \
  --ghsl ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --context-output ~/data/outputs/baseline_context.png
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
    p = argparse.ArgumentParser(description="Plot comparison of three baselines plus GHSL 10 m reference.")
    p.add_argument("--baseline0", required=True, help="Path to WSF-uniform baseline raster")
    p.add_argument("--baseline1", required=True, help="Path to embeddings-only baseline raster")
    p.add_argument("--baseline2", required=True, help="Path to embeddings+WSF baseline raster")
    p.add_argument("--ghsl10m", required=True, help="Path to GHSL 2019 10 m raster")
    p.add_argument("--output", required=True, help="Output PNG for 2x4 baseline comparison")
    p.add_argument("--agg-factor", type=int, default=10, help="Aggregation factor for lower row")
    p.add_argument("--title", default="Baseline comparison", help="Figure title")
    p.add_argument("--ghsl", default=None, help="Optional GHSL coarse raster for context figure")
    p.add_argument("--wsf", default=None, help="Optional WSF binary raster for context figure")
    p.add_argument("--context-output", default=None, help="Optional output PNG for context figure")
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


def get_extent(profile: dict) -> Tuple[float, float, float, float]:
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return (left, right, bottom, top)


def aggregated_extent(profile: dict, factor: int) -> Tuple[float, float, float, float]:
    transform = profile["transform"]
    width = (profile["width"] // factor)
    height = (profile["height"] // factor)
    left = transform.c
    top = transform.f
    right = left + transform.a * factor * width
    bottom = top + transform.e * factor * height
    return (left, right, bottom, top)


def robust_limits(arrays, low=2, high=98):
    vals = []
    for arr in arrays:
        x = arr[np.isfinite(arr)]
        if x.size:
            vals.append(x)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    vmin = float(np.percentile(vals, low))
    vmax = float(np.percentile(vals, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def add_image(ax, arr, extent, title, cmap, vmin=None, vmax=None, cbar_label=None):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    return im


def main():
    args = parse_args()

    b0_path = Path(args.baseline0).expanduser().resolve()
    b1_path = Path(args.baseline1).expanduser().resolve()
    b2_path = Path(args.baseline2).expanduser().resolve()
    ghsl10m_path = Path(args.ghsl10m).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    b0, p0 = read_single_band(b0_path)
    b1, p1 = read_single_band(b1_path)
    b2, p2 = read_single_band(b2_path)

    ensure_same_grid(p0, p1, "baseline0", "baseline1")
    ensure_same_grid(p0, p2, "baseline0", "baseline2")

    ghsl10 = align_to_target(ghsl10m_path, p0, Resampling.bilinear)

    fine_extent = get_extent(p0)

    agg_factor = int(args.agg_factor)
    if agg_factor < 1:
        raise ValueError("--agg-factor must be >= 1")

    b0_agg = aggregate_mean(b0, agg_factor)
    b1_agg = aggregate_mean(b1, agg_factor)
    b2_agg = aggregate_mean(b2, agg_factor)
    ghsl10_agg = aggregate_mean(ghsl10, agg_factor)
    agg_extent = aggregated_extent(p0, agg_factor)

    fine_vmin, fine_vmax = robust_limits([b0, b1, b2, ghsl10], low=2, high=98)
    agg_vmin, agg_vmax = robust_limits([b0_agg, b1_agg, b2_agg, ghsl10_agg], low=2, high=98)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    add_image(
        axes[0, 0], b0, fine_extent,
        title="A. Baseline 0: WSF-uniform\n(m² per 10 m pixel)",
        cmap="magma", vmin=fine_vmin, vmax=fine_vmax,
        cbar_label="Allocated built-up (m² per 10 m pixel)"
    )
    add_image(
        axes[0, 1], b1, fine_extent,
        title="B. Baseline 1: embeddings-only\n(m² per 10 m pixel)",
        cmap="magma", vmin=fine_vmin, vmax=fine_vmax,
        cbar_label="Allocated built-up (m² per 10 m pixel)"
    )
    add_image(
        axes[0, 2], b2, fine_extent,
        title="C. Baseline 2: embeddings + WSF\n(m² per 10 m pixel)",
        cmap="magma", vmin=fine_vmin, vmax=fine_vmax,
        cbar_label="Allocated built-up (m² per 10 m pixel)"
    )
    add_image(
        axes[0, 3], ghsl10, fine_extent,
        title="D. GHSL 2019 10 m reference\n(m² per 10 m pixel)",
        cmap="magma", vmin=fine_vmin, vmax=fine_vmax,
        cbar_label="Built-up surface (m² per 10 m pixel)"
    )

    add_image(
        axes[1, 0], b0_agg, agg_extent,
        title=f"E. Baseline 0 aggregated ({agg_factor}×{agg_factor})\n(mean m² per 10 m pixel)",
        cmap="magma", vmin=agg_vmin, vmax=agg_vmax,
        cbar_label="Mean allocated built-up (m² per 10 m pixel)"
    )
    add_image(
        axes[1, 1], b1_agg, agg_extent,
        title=f"F. Baseline 1 aggregated ({agg_factor}×{agg_factor})\n(mean m² per 10 m pixel)",
        cmap="magma", vmin=agg_vmin, vmax=agg_vmax,
        cbar_label="Mean allocated built-up (m² per 10 m pixel)"
    )
    add_image(
        axes[1, 2], b2_agg, agg_extent,
        title=f"G. Baseline 2 aggregated ({agg_factor}×{agg_factor})\n(mean m² per 10 m pixel)",
        cmap="magma", vmin=agg_vmin, vmax=agg_vmax,
        cbar_label="Mean allocated built-up (m² per 10 m pixel)"
    )
    add_image(
        axes[1, 3], ghsl10_agg, agg_extent,
        title=f"H. GHSL 2019 10 m aggregated ({agg_factor}×{agg_factor})\n(mean m² per 10 m pixel)",
        cmap="magma", vmin=agg_vmin, vmax=agg_vmax,
        cbar_label="Mean built-up surface (m² per 10 m pixel)"
    )

    fig.suptitle(args.title, fontsize=22)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved baseline comparison figure to: {out_path}")

    if args.ghsl and args.wsf and args.context_output:
        ghsl_path = Path(args.ghsl).expanduser().resolve()
        wsf_path = Path(args.wsf).expanduser().resolve()
        context_out = Path(args.context_output).expanduser().resolve()

        ghsl_fine = align_to_target(ghsl_path, p0, Resampling.nearest)
        wsf = align_to_target(wsf_path, p0, Resampling.nearest)

        g_vmin, g_vmax = robust_limits([ghsl_fine], low=2, high=98)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        add_image(
            axes2[0], ghsl_fine, fine_extent,
            title="A. GHSL coarse total built-up\n(m² per 1 km cell)",
            cmap="viridis", vmin=g_vmin, vmax=g_vmax,
            cbar_label="Built-up surface (m² per 1 km cell)"
        )
        add_image(
            axes2[1], wsf, fine_extent,
            title="B. WSF settlement mask (0/1)",
            cmap="Greys", vmin=0, vmax=1,
            cbar_label="WSF support (0/1)"
        )

        fig2.suptitle("Context layers", fontsize=20)
        context_out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(context_out, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"[INFO] Saved context figure to: {context_out}")


if __name__ == "__main__":
    main()
