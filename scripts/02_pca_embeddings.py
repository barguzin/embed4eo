#!/usr/bin/env python3
"""Fit IncrementalPCA on a sample of a multiband embedding raster and write
an 8-band PCA-transformed raster.

Designed for large Earth observation embedding mosaics that do not fit
comfortably in memory. The script:
1. samples valid pixels from the input raster using reservoir sampling;
2. fits IncrementalPCA on the sample;
3. transforms the full raster window-by-window;
4. writes the reduced raster and a small JSON report.

Example
-------
python 02_pca_embeddings.py \
  --input ~/data/aef_accra_2019/mosaic_accra_2019.tiff \
  --output ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --model ~/data/aef_accra_2019/mosaic_accra_2019_pca8.joblib \
  --report ~/data/aef_accra_2019/mosaic_accra_2019_pca8_report.json \
  --n-components 8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import profile
from typing import Iterable, Iterator, List, Tuple

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.decomposition import IncrementalPCA


DEFAULT_WINDOW = 512
DEFAULT_SAMPLE_PIXELS = 250_000
DEFAULT_BATCH_SIZE = 20_000
DEFAULT_RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to multiband embedding raster")
    parser.add_argument("--output", required=True, help="Path to output PCA raster")
    parser.add_argument("--model", required=True, help="Path to save fitted PCA model (.joblib)")
    parser.add_argument("--report", required=True, help="Path to save JSON report")
    parser.add_argument("--n-components", type=int, default=8, help="Number of PCA components")
    parser.add_argument(
        "--sample-pixels",
        type=int,
        default=DEFAULT_SAMPLE_PIXELS,
        help="Target number of valid pixels to sample for PCA fitting",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW,
        help="Window size (pixels) for streaming reads",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for IncrementalPCA partial fits",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducible sampling",
    )
    return parser.parse_args()


def make_windows(width: int, height: int, size: int) -> Iterator[Window]:
    for row_off in range(0, height, size):
        h = min(size, height - row_off)
        for col_off in range(0, width, size):
            w = min(size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


class ReservoirSampler:
    """Fixed-size reservoir sampler for rows of a 2D numpy array."""

    def __init__(self, capacity: int, n_features: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.n_features = int(n_features)
        self.rng = rng
        self.data = np.empty((self.capacity, self.n_features), dtype=np.float32)
        self.size = 0
        self.seen = 0

    def update(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        if rows.ndim != 2 or rows.shape[1] != self.n_features:
            raise ValueError("rows must be 2D with n_features columns")

        for row in rows:
            self.seen += 1
            if self.size < self.capacity:
                self.data[self.size] = row
                self.size += 1
            else:
                j = self.rng.integers(0, self.seen)
                if j < self.capacity:
                    self.data[j] = row

    def get(self) -> np.ndarray:
        return self.data[: self.size].copy()


def reshape_valid_pixels(arr: np.ndarray) -> np.ndarray:
    """Convert (bands, rows, cols) to (n_valid_pixels, bands)."""
    bands, rows, cols = arr.shape
    flat = np.moveaxis(arr, 0, -1).reshape(rows * cols, bands)
    valid_mask = np.all(np.isfinite(flat), axis=1)
    return flat[valid_mask].astype(np.float32, copy=False)


def sample_pixels(
    src: rasterio.DatasetReader,
    sample_pixels: int,
    window_size: int,
    random_state: int,
) -> Tuple[np.ndarray, int]:
    rng = np.random.default_rng(random_state)
    sampler = ReservoirSampler(capacity=sample_pixels, n_features=src.count, rng=rng)

    for window in make_windows(src.width, src.height, window_size):
        arr = src.read(window=window, out_dtype="float32")
        rows = reshape_valid_pixels(arr)
        sampler.update(rows)

    return sampler.get(), sampler.seen


def fit_ipca(sample: np.ndarray, n_components: int, batch_size: int) -> IncrementalPCA:
    if sample.shape[0] < n_components:
        raise ValueError(
            f"Sample has only {sample.shape[0]} rows but n_components={n_components}."
        )

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for start in range(0, sample.shape[0], batch_size):
        stop = min(start + batch_size, sample.shape[0])
        ipca.partial_fit(sample[start:stop])
    return ipca


def transform_full_raster(
    src: rasterio.DatasetReader,
    dst_path: Path,
    ipca: IncrementalPCA,
    window_size: int,
) -> None:
    profile = src.profile.copy()

    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    profile.update(
        count=ipca.n_components_,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        dst_path.unlink()

    with rasterio.open(dst_path, "w", **profile) as dst:
        for window in make_windows(src.width, src.height, window_size):
            arr = src.read(window=window, out_dtype="float32")
            bands, rows, cols = arr.shape
            flat = np.moveaxis(arr, 0, -1).reshape(rows * cols, bands)
            valid_mask = np.all(np.isfinite(flat), axis=1)

            out = np.full((rows * cols, ipca.n_components_), np.nan, dtype=np.float32)
            if np.any(valid_mask):
                transformed = ipca.transform(flat[valid_mask])
                out[valid_mask] = transformed.astype(np.float32, copy=False)

            out = out.reshape(rows, cols, ipca.n_components_)
            out = np.moveaxis(out, -1, 0)
            dst.write(out, window=window)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    with rasterio.open(input_path) as src:
        print(f"[INFO] Input raster: {input_path}")
        print(f"[INFO] Shape: bands={src.count}, height={src.height}, width={src.width}")
        print(f"[INFO] Sampling up to {args.sample_pixels:,} valid pixels for PCA fit")

        sample, total_valid = sample_pixels(
            src,
            sample_pixels=args.sample_pixels,
            window_size=args.window_size,
            random_state=args.random_state,
        )
        print(f"[INFO] Total valid pixels seen: {total_valid:,}")
        print(f"[INFO] Sample retained: {sample.shape[0]:,}")

        ipca = fit_ipca(
            sample=sample,
            n_components=args.n_components,
            batch_size=args.batch_size,
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(ipca, model_path)
        print(f"[INFO] Saved PCA model to: {model_path}")

        transform_full_raster(
            src=src,
            dst_path=output_path,
            ipca=ipca,
            window_size=args.window_size,
        )
        print(f"[INFO] Saved PCA raster to: {output_path}")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "model": str(model_path),
        "n_components": int(ipca.n_components_),
        "sample_pixels_target": int(args.sample_pixels),
        "sample_pixels_used": int(sample.shape[0]),
        "total_valid_pixels_seen": int(total_valid),
        "window_size": int(args.window_size),
        "batch_size": int(args.batch_size),
        "random_state": int(args.random_state),
        "explained_variance_ratio": [float(x) for x in ipca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(np.sum(ipca.explained_variance_ratio_)),
        "singular_values": [float(x) for x in ipca.singular_values_],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[INFO] Saved PCA report to: {report_path}")
    print(f"[INFO] Explained variance (sum): {report['explained_variance_ratio_sum']:.4f}")


if __name__ == "__main__":
    main()
