#!/usr/bin/env python3
"""
05_plot_four_panel.py

Create a four-panel visualization for the WSF-uniform baseline:
1) coarse GHSL built-up surface
2) WSF binary mask
3) fine 10 m baseline prediction
4) baseline prediction aggregated to ~100 m

Optional: overlay fallback coarse cells if a fallback raster is provided.

Example
-------
python 05_plot_four_panel.py \
  --ghsl ~/data/GHSL_BUILD/cropped_ghsl.tif \
  --wsf ~/data/WSF_Data/cropped_wsf.tif \
  --baseline ~/data/outputs/wsf_uniform_baseline.tif \
  --output ~/data/outputs/wsf_uniform_four_panel.png \
  --fallback ~/data/outputs/wsf_uniform_baseline_fallback_cells.tif \
  --agg-factor 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject


# -----------------------------
# parsing
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create four-panel baseline visualization")
    p.add_argument("--ghsl", required=True, help="Path to coarse GHSL raster")
    p.add_argument("--wsf", required=True, help="Path to fine WSF raster")
    p.add_argument("--baseline", required=True, help="Path to fine baseline raster")
    p.add_argument("--output", required=True, help="Path to output figure (png/pdf)")
    p.add_argument(
        "--fallback",
        default=None,
        help="Optional fallback coarse-cell raster projected to fine grid",
    )
    p.add_argument(
        "--agg-factor",
        type=int,
        default=10,
        help="Aggregation factor for panel 4 (default: 10 => ~100 m if fine grid is ~10 m)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (default: 300)",
    )
    p.add_argument(
        "--title",
        default="WSF-uniform baseline",
        help="Optional figure supertitle",
    )
    return p.parse_args()


# -----------------------------
# raster helpers
# -----------------------------

def read_first_band(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        profile["transform"] = src.transform
        profile["crs"] = src.crs
        profile["nodata"] = src.nodata
        profile["width"] = src.width
        profile["height"] = src.height
    return arr, profile


def masked_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float32, copy=False)
    if nodata is not None and np.isfinite(nodata):
        out = np.where(arr == nodata, np.nan, out)
    return out


def raster_extent(transform, height: int, width: int) -> Tuple[float, float, float, float]:
    bottom, left, top, right = None, None, None, None
    # array_bounds returns (south, west, north, east)
    south, west, north, east = array_bounds(height, width, transform)
    return (west, east, south, north)


def reproject_to_match(
    src_arr: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    resampling: Resampling,
    dst_nodata: float,
) -> np.ndarray:
    dst = np.full((dst_profile["height"], dst_profile["width"]), dst_nodata, dtype=np.float32)

    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=src_profile.get("nodata", None),
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return dst


def aggregate_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate 2D array by block mean, ignoring NaN."""
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor
    arr = arr[:h2, :w2]

    reshaped = arr.reshape(h2 // factor, factor, w2 // factor, factor)
    with np.errstate(invalid="ignore"):
        return np.nanmean(reshaped, axis=(1, 3))


def aggregate_transform(transform, factor: int):
    from affine import Affine
    return transform * Affine.scale(factor, factor)


def finite_quantiles(arr: np.ndarray, qlo: float = 0.02, qhi: float = 0.98) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(vals, [qlo, qhi])
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def add_image(ax, arr, extent, title, cmap, vmin=None, vmax=None, cbar_label=None):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar_label is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return im


def overlay_fallback(ax, fallback_arr: Optional[np.ndarray], extent):
    if fallback_arr is None:
        return
    mask = np.where(np.isfinite(fallback_arr) & (fallback_arr > 0), 1.0, np.nan)
    # lightly shade fallback footprint
    ax.imshow(mask, extent=extent, origin="upper", cmap="Reds", alpha=0.28, vmin=0, vmax=1)


# -----------------------------
# main
# -----------------------------

def main() -> None:
    args = parse_args()

    ghsl_path = Path(args.ghsl).expanduser().resolve()
    wsf_path = Path(args.wsf).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    fallback_path = Path(args.fallback).expanduser().resolve() if args.fallback else None

    wsf, wsf_prof = read_first_band(wsf_path)
    baseline, base_prof = read_first_band(baseline_path)

    # Fine grid for main plotting panels
    fine_extent = raster_extent(base_prof["transform"], base_prof["height"], base_prof["width"])

    # WSF and baseline should already align, but mask safely
    wsf = masked_array(wsf, wsf_prof.get("nodata"))
    baseline = masked_array(baseline, base_prof.get("nodata"))

    # Reproject GHSL to fine grid only for visualization consistency
    ghsl, ghsl_prof = read_first_band(ghsl_path)
    ghsl_fine = reproject_to_match(
        src_arr=ghsl.astype(np.float32, copy=False),
        src_profile=ghsl_prof,
        dst_profile=base_prof,
        resampling=Resampling.nearest,
        dst_nodata=np.nan,
    )

    fallback_fine = None
    if fallback_path is not None:
        fallback_fine, fall_prof = read_first_band(fallback_path)
        fallback_fine = masked_array(fallback_fine, fall_prof.get("nodata"))
        if (
            fall_prof["height"] != base_prof["height"]
            or fall_prof["width"] != base_prof["width"]
            or fall_prof["transform"] != base_prof["transform"]
            or fall_prof["crs"] != base_prof["crs"]
        ):
            fallback_fine = reproject_to_match(
                src_arr=fallback_fine,
                src_profile=fall_prof,
                dst_profile=base_prof,
                resampling=Resampling.nearest,
                dst_nodata=np.nan,
            )

    # Aggregated baseline (~100 m if fine grid is 10 m and factor is 10)
    agg_factor = max(1, int(args.agg_factor))
    baseline_agg = aggregate_mean(baseline, agg_factor)
    agg_transform = aggregate_transform(base_prof["transform"], agg_factor)
    agg_extent = raster_extent(agg_transform, baseline_agg.shape[0], baseline_agg.shape[1])

    # Robust color scaling for baseline maps
    b_lo, b_hi = finite_quantiles(baseline, 0.02, 0.98)
    a_lo, a_hi = finite_quantiles(baseline_agg, 0.02, 0.98)
    g_lo, g_hi = finite_quantiles(ghsl_fine, 0.02, 0.98)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    add_image(
        axes[0], ghsl_fine, fine_extent,
        title="A. GHSL coarse built-up (visualized on fine grid)",
        # cmap="viridis", vmin=g_lo, vmax=g_hi, cbar_label="Built-up surface"
        cmap="viridis", vmin=g_lo, vmax=g_hi, cbar_label="Built-up surface (m² per 1 km cell)"
    )
    overlay_fallback(axes[0], fallback_fine, fine_extent)

    add_image(
        axes[1], wsf, fine_extent,
        title="B. WSF binary support mask",
        cmap="Greys", vmin=0, vmax=1, cbar_label="WSF"
    )
    overlay_fallback(axes[1], fallback_fine, fine_extent)

    add_image(
        axes[2], baseline, fine_extent,
        title="C. Uniform-over-WSF baseline (fine grid)",
        # cmap="magma", vmin=b_lo, vmax=b_hi, cbar_label="Allocated built-up"
        cmap="magma", vmin=b_lo, vmax=b_hi, cbar_label="Allocated built-up (m² per 10 m pixel)"
    )
    overlay_fallback(axes[2], fallback_fine, fine_extent)

    add_image(
        axes[3], baseline_agg, agg_extent,
        title=f"D. Baseline aggregated by factor {agg_factor}",
        # cmap="magma", vmin=a_lo, vmax=a_hi, cbar_label="Mean allocated built-up"
        cmap="magma", vmin=a_lo, vmax=a_hi, cbar_label="Mean allocated built-up (m² per 10 m pixel)"
    )

    if args.title:
        fig.suptitle(args.title, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved figure to: {output_path}")
    print(f"[INFO] Fine extent: {fine_extent}")
    print(f"[INFO] Aggregation factor: {agg_factor}")
    if fallback_fine is not None:
        n_fb = int(np.nansum(fallback_fine > 0))
        print(f"[INFO] Fallback fine pixels highlighted: {n_fb:,}")


if __name__ == "__main__":
    main()
