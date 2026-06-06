#!/usr/bin/env python3
"""
01_figure1_poster.py

Prepare the four-panel Figure 1 poster input/context figure.

Panels
------
A. Study Area
B. Coarse GHSL
C. World Settlement Footprint
D. EO Embeddings

Example
-------
mamba run -n diss python scripts/01_figure1_poster.py \
  --output-dir ~/data/outputs
"""

from __future__ import annotations

import argparse
import math
import urllib.request
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create poster Figure 1 with study area and input layers.")
    p.add_argument("--rgb", default=None, help="Optional local satellite/RGB basemap raster.")
    p.add_argument("--basemap-zoom", type=int, default=14, help="Web satellite tile zoom used when --rgb is omitted.")
    p.add_argument("--context-basemap-zoom", type=int, default=11, help="Web satellite tile zoom for the wider Panel A context.")
    p.add_argument("--tile-cache", default="~/data/outputs/basemap_cache", help="Directory for cached web satellite tiles.")
    p.add_argument("--aoi", default="~/data/aux/bbox_accra_dissolve.gpkg", help="AOI vector used in preprocessing.")
    p.add_argument(
        "--context-aoi",
        default="~/Downloads/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg",
        help="Vector layer used to set the wider Panel A context extent.",
    )
    p.add_argument("--context-name", default="Accra", help="Name filter for --context-aoi, e.g. Accra.")
    p.add_argument("--context-pad", type=float, default=0.08, help="Fractional padding around the Panel A context extent.")
    p.add_argument("--ghsl", default="~/data/GHSL_BUILD/cropped_ghsl.tif", help="Cropped coarse GHSL 1 km raster.")
    p.add_argument("--wsf", default="~/data/WSF_Data/cropped_wsf.tif", help="Cropped WSF raster.")
    p.add_argument("--embeddings", default="~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif", help="EO embeddings raster.")
    p.add_argument("--embedding-bands", type=int, nargs=3, default=[1, 2, 3], help="Three 1-based embedding bands to display as RGB.")
    p.add_argument("--output-dir", default="~/data/outputs", help="Directory for poster figure exports.")
    p.add_argument("--output", default=None, help="Optional PNG output path.")
    p.add_argument("--svg-output", default=None, help="Optional SVG output path.")
    p.add_argument("--pdf-output", default=None, help="Optional PDF output path.")
    return p.parse_args()


def read_profile(path: Path) -> dict:
    with rasterio.open(path) as src:
        return src.profile.copy()


def raster_extent(profile: dict) -> Tuple[float, float, float, float]:
    transform = profile["transform"]
    width = profile["width"]
    height = profile["height"]
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return (left, right, bottom, top)


def bounds_to_extent(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    return left, right, bottom, top


def extent_to_bounds(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, right, bottom, top = extent
    return left, bottom, right, top


def read_single(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def read_rgb(path: Path, bands: Iterable[int] = (1, 2, 3)) -> tuple[np.ndarray, dict]:
    band_list = list(bands)
    with rasterio.open(path) as src:
        band_count = src.count
        if max(band_list) > band_count:
            raise ValueError(f"{path} has {band_count} bands, but requested bands {band_list}")
        arr = src.read(band_list).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def normalize_rgb(arr: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    out = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
    for i in range(3):
        band = arr[i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        vmin = float(np.percentile(finite, low))
        vmax = float(np.percentile(finite, high))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
        out[:, :, i] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
    out[~np.isfinite(out)] = 0
    return out


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_mercator_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    half_world = 20037508.342789244
    n = 2 ** zoom
    tile_size = 2 * half_world / n
    left = -half_world + x * tile_size
    right = left + tile_size
    top = half_world - y * tile_size
    bottom = top - tile_size
    return left, bottom, right, top


def fetch_esri_tile(x: int, y: int, zoom: int, cache_dir: Path) -> Image.Image:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tile_path = cache_dir / f"esri_world_imagery_z{zoom}_x{x}_y{y}.jpg"
    if not tile_path.exists():
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
        with urllib.request.urlopen(url, timeout=30) as response:
            tile_path.write_bytes(response.read())
    return Image.open(tile_path).convert("RGB")


def padded_bounds_for_aspect(
    bounds: tuple[float, float, float, float],
    aspect: float,
    pad_fraction: float,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    width = right - left
    height = top - bottom
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounds: {bounds}")

    current_aspect = width / height
    if current_aspect > aspect:
        new_width = width
        new_height = width / aspect
    else:
        new_height = height
        new_width = height * aspect

    new_width *= 1.0 + pad_fraction * 2.0
    new_height *= 1.0 + pad_fraction * 2.0
    cx = (left + right) / 2.0
    cy = (bottom + top) / 2.0
    return (
        cx - new_width / 2.0,
        cy - new_height / 2.0,
        cx + new_width / 2.0,
        cy + new_height / 2.0,
    )


def profile_for_bounds(target_profile: dict, bounds: tuple[float, float, float, float]) -> dict:
    profile = target_profile.copy()
    profile["transform"] = from_bounds(
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        target_profile["width"],
        target_profile["height"],
    )
    return profile


def fetch_web_satellite_basemap(target_profile: dict, zoom: int, cache_dir: Path) -> np.ndarray:
    from pyproj import Transformer

    extent = raster_extent(target_profile)
    left, right, bottom, top = extent
    to_lonlat = Transformer.from_crs(target_profile["crs"], "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_lonlat.transform(left, bottom)
    lon_max, lat_max = to_lonlat.transform(right, top)

    x_min, y_max = lonlat_to_tile(lon_min, lat_min, zoom)
    x_max, y_min = lonlat_to_tile(lon_max, lat_max, zoom)
    x0, x1 = sorted((x_min, x_max))
    y0, y1 = sorted((y_min, y_max))

    tile_size_px = 256
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * tile_size_px, (y1 - y0 + 1) * tile_size_px))
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            tile = fetch_esri_tile(x, y, zoom, cache_dir)
            mosaic.paste(tile, ((x - x0) * tile_size_px, (y - y0) * tile_size_px))

    left_m, _, _, top_m = tile_mercator_bounds(x0, y0, zoom)
    _, bottom_m, right_m, _ = tile_mercator_bounds(x1, y1, zoom)
    src_transform = from_bounds(left_m, bottom_m, right_m, top_m, mosaic.width, mosaic.height)
    src_arr = np.asarray(mosaic).astype(np.uint8)

    dst = np.zeros((3, target_profile["height"], target_profile["width"]), dtype=np.uint8)
    for i in range(3):
        reproject(
            source=src_arr[:, :, i],
            destination=dst[i],
            src_transform=src_transform,
            src_crs="EPSG:3857",
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            resampling=Resampling.bilinear,
        )
    return np.moveaxis(dst, 0, -1).astype(np.float32) / 255.0


def align_single_to_target(src_path: Path, target_profile: dict, resampling: Resampling) -> np.ndarray:
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


def align_rgb_to_target(src_path: Path, target_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        count = min(src.count, 3)
        if count < 3:
            raise ValueError(f"{src_path} must have at least 3 bands to be used as RGB")
        src_arr = src.read([1, 2, 3]).astype(np.float32, copy=False)
        dst = np.full((3, target_profile["height"], target_profile["width"]), np.nan, dtype=np.float32)
        src_nodata = src.nodata if src.nodata is not None else np.nan
        for i in range(3):
            reproject(
                source=src_arr[i],
                destination=dst[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile["transform"],
                dst_crs=target_profile["crs"],
                src_nodata=src_nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return dst


def vector_bounds_in_crs(
    vector_path: Path,
    dst_crs,
    name_filter: str | None = None,
) -> tuple[float, float, float, float] | None:
    if not vector_path.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None
    gdf = gpd.read_file(vector_path)
    if name_filter:
        attrs = gdf[[c for c in gdf.columns if c != "geometry"]].astype(str)
        mask = attrs.apply(lambda row: row.str.contains(name_filter, case=False, na=False).any(), axis=1)
        filtered = gdf[mask]
        if not filtered.empty:
            gdf = filtered
    if gdf.empty:
        return None
    if gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def add_bounds_box(ax, bounds: tuple[float, float, float, float], color: str = "#ff8c00") -> None:
    minx, miny, maxx, maxy = bounds
    rect = Rectangle(
        (minx, miny),
        maxx - minx,
        maxy - miny,
        fill=False,
        edgecolor=color,
        linewidth=3.5,
        joinstyle="miter",
    )
    ax.add_patch(rect)


def robust_limits(arr: np.ndarray, low: float = 2, high: float = 98) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def resolve_outputs(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    png_path = Path(args.output).expanduser().resolve() if args.output else output_dir / "figure1_inputs_poster.png"
    svg_path = Path(args.svg_output).expanduser().resolve() if args.svg_output else output_dir / "figure1_inputs_poster.svg"
    pdf_path = Path(args.pdf_output).expanduser().resolve() if args.pdf_output else None
    return png_path, svg_path, pdf_path


def main() -> None:
    args = parse_args()

    ghsl_path = Path(args.ghsl).expanduser().resolve()
    wsf_path = Path(args.wsf).expanduser().resolve()
    embeddings_path = Path(args.embeddings).expanduser().resolve()
    aoi_path = Path(args.aoi).expanduser().resolve()
    context_aoi_path = Path(args.context_aoi).expanduser().resolve()
    png_path, svg_path, pdf_path = resolve_outputs(args)

    embed_rgb_raw, embed_profile = read_rgb(embeddings_path, args.embedding_bands)
    embed_rgb = normalize_rgb(embed_rgb_raw)
    target_extent = raster_extent(embed_profile)
    panel_aspect = embed_profile["width"] / embed_profile["height"]

    aoi_bounds = vector_bounds_in_crs(aoi_path, embed_profile["crs"]) or extent_to_bounds(target_extent)
    context_bounds_raw = vector_bounds_in_crs(
        context_aoi_path,
        embed_profile["crs"],
        name_filter=args.context_name,
    )
    context_extent = padded_bounds_for_aspect(
        context_bounds_raw or aoi_bounds,
        aspect=panel_aspect,
        pad_fraction=args.context_pad,
    )
    context_profile = profile_for_bounds(embed_profile, context_extent)
    context_plot_extent = bounds_to_extent(context_extent)

    if args.rgb:
        basemap_raw = align_rgb_to_target(Path(args.rgb).expanduser().resolve(), embed_profile)
        basemap_rgb = normalize_rgb(basemap_raw)
        context_basemap_rgb = fetch_web_satellite_basemap(
            context_profile,
            zoom=args.context_basemap_zoom,
            cache_dir=Path(args.tile_cache).expanduser().resolve(),
        )
    else:
        basemap_rgb = fetch_web_satellite_basemap(
            embed_profile,
            zoom=args.basemap_zoom,
            cache_dir=Path(args.tile_cache).expanduser().resolve(),
        )
        context_basemap_rgb = fetch_web_satellite_basemap(
            context_profile,
            zoom=args.context_basemap_zoom,
            cache_dir=Path(args.tile_cache).expanduser().resolve(),
        )

    ghsl, ghsl_profile = read_single(ghsl_path)
    wsf = align_single_to_target(wsf_path, embed_profile, Resampling.nearest)
    wsf_mask = np.where(np.isfinite(wsf) & (wsf > 0), 1.0, np.nan)

    ghsl_extent = raster_extent(ghsl_profile)
    ghsl_vmin, ghsl_vmax = robust_limits(ghsl, low=2, high=98)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.8), constrained_layout=True)
    titles = [
        "A. Study Area",
        "B. World Settlement Footprint",
        "C. Coarse GHSL",
        "D. EO Embeddings",
    ]

    axes[0].imshow(context_basemap_rgb, extent=context_plot_extent, origin="upper", interpolation="nearest")
    add_bounds_box(axes[0], aoi_bounds)

    axes[1].imshow(basemap_rgb, extent=target_extent, origin="upper", interpolation="nearest")
    axes[1].imshow(
        wsf_mask,
        extent=target_extent,
        origin="upper",
        cmap=ListedColormap(["#000000"]),
        alpha=0.5,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    axes[2].imshow(basemap_rgb, extent=target_extent, origin="upper", interpolation="nearest")
    axes[2].imshow(
        ghsl,
        extent=ghsl_extent,
        origin="upper",
        cmap="magma",
        vmin=ghsl_vmin,
        vmax=ghsl_vmax,
        alpha=0.5,
        interpolation="nearest",
    )

    axes[3].imshow(basemap_rgb, extent=target_extent, origin="upper", interpolation="nearest")
    axes[3].imshow(embed_rgb, extent=target_extent, origin="upper", alpha=0.5, interpolation="nearest")

    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        extent = context_plot_extent if i == 0 else target_extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    for path in [png_path, svg_path, pdf_path]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved PNG to: {png_path}")
    print(f"[INFO] Saved SVG to: {svg_path}")
    if pdf_path is not None:
        print(f"[INFO] Saved PDF to: {pdf_path}")
    if not args.rgb:
        print("[INFO] No --rgb basemap supplied; fetched Esri World Imagery satellite tiles.")


if __name__ == "__main__":
    main()
