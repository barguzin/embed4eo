#!/usr/bin/env python3
"""
15_train_residual_dilated_wsf_diffnorm.py

Baseline 2c: dilated residual embeddings + WSF differentiable mass allocator.

Inputs
------
- PCA raster (e.g. 8-band embedding PCA GeoTIFF)
- WSF-derived feature raster (bands like wsf_bin, wsf_dens_5, wsf_dens_11, wsf_dist)
- fine-grid coarse cell-id raster
- lookup CSV with coarse targets (use ghsl_value_adj by default)

Model
-----
Dilated residual fully convolutional network with a positive score output
(Softplus).

Loss
----
The CNN predicts positive raw scores from random initialization. Scores are
differentiably normalized within each coarse cell and multiplied by that cell's
GHSL target. Training uses the resulting mass-preserving fine prediction:

total_loss = raw_coarse_weight * raw_coarse_loss
             + tv_weight * tv_loss
             + edge_tv_weight * edge_aware_tv_loss
             + boundary_weight * coarse_boundary_loss
             + wsf_weight * outside_wsf_fraction

where:
- raw_coarse_loss compares aggregated raw scores to coarse GHSL targets
- tv_loss is total variation on the normalized fine prediction
- edge_aware_tv_loss is total variation downweighted across feature edges
- coarse_boundary_loss penalizes jumps across adjacent GHSL coarse-cell borders
- outside_wsf_fraction penalizes normalized predicted mass outside WSF support

Outputs
-------
- raw fine prediction raster
- renormalized fine prediction raster (exact coarse mass preservation)
- JSON report
- optional loss plot
- optional model weights

Example
-------
python 15_train_residual_dilated_wsf_diffnorm.py \
  --pca ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --wsf-features ~/data/WSF_Data/cropped_wsf_features.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --value-column ghsl_value_adj \
  --pred-out ~/data/outputs/dilated_wsf_diffnorm_raw.tif \
  --pred-norm-out ~/data/outputs/dilated_wsf_diffnorm_norm.tif \
  --report ~/data/outputs/dilated_wsf_diffnorm_report.json \
  --loss-plot ~/data/outputs/dilated_wsf_diffnorm_loss.png \
  --model-out ~/data/outputs/dilated_wsf_diffnorm_model.pt \
  --epochs 500 \
  --lr 1e-3 \
  --tv-weight 0.0 \
  --edge-tv-weight 5e-5 \
  --edge-tv-alpha 5.0 \
  --boundary-weight 1e-4 \
  --wsf-weight 0.02 \
  --raw-coarse-weight 1.0 \
  --score-floor 1e-6 \
  --hidden 32 \
  --depth 4 \
  --dilations 1 2 4 8 \
  --wsf-band 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train dilated residual embeddings + WSF diffnorm allocator.")
    p.add_argument("--pca", required=True, help="Path to PCA raster")
    p.add_argument("--wsf-features", required=True, help="Path to WSF feature raster")
    p.add_argument("--cell-ids", required=True, help="Path to fine-grid coarse cell-id raster")
    p.add_argument("--lookup", required=True, help="Path to coarse lookup CSV")
    p.add_argument("--value-column", default="ghsl_value_adj", help="Lookup column for coarse targets")
    p.add_argument("--wsf-band", type=int, default=1, help="1-based band index in WSF feature raster for binary WSF mask")
    p.add_argument("--pred-out", required=True, help="Output raw fine prediction raster")
    p.add_argument("--pred-norm-out", required=True, help="Output renormalized fine prediction raster")
    p.add_argument("--report", required=True, help="Output JSON report")
    p.add_argument("--loss-plot", default=None, help="Optional output PNG for training loss")
    p.add_argument("--model-out", default=None, help="Optional output path for model weights")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--tv-weight", type=float, default=1e-5)
    p.add_argument(
        "--edge-tv-weight",
        type=float,
        default=0.0,
        help="Weight for feature-edge-aware TV on the normalized prediction",
    )
    p.add_argument(
        "--edge-tv-alpha",
        type=float,
        default=5.0,
        help="Strength of feature-gradient downweighting for edge-aware TV",
    )
    p.add_argument(
        "--boundary-weight",
        type=float,
        default=0.0,
        help="Weight for continuity loss across neighboring pixels in different coarse GHSL cells",
    )
    p.add_argument("--wsf-weight", type=float, default=0.05)
    p.add_argument(
        "--raw-coarse-weight",
        type=float,
        default=1.0,
        help="Weight for scaled coarse-cell MSE on the raw score surface",
    )
    p.add_argument("--score-floor", type=float, default=1e-6, help="Minimum raw score used inside cell normalization")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument(
        "--dilations",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Dilation sequence cycled through residual blocks",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lookup(csv_path: Path, value_column: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if value_column not in reader.fieldnames:
            raise ValueError(f"Column '{value_column}' not found in lookup CSV: {reader.fieldnames}")
        if "cell_id" not in reader.fieldnames:
            raise ValueError("Lookup CSV must contain 'cell_id'")
        for row in reader:
            cid = int(row["cell_id"])
            val = float(row[value_column])
            out[cid] = val
    return out


def read_rasters(
    pca_path: Path,
    wsf_path: Path,
    cell_ids_path: Path,
    wsf_band_1based: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    with (
        rasterio.open(pca_path) as src_pca,
        rasterio.open(wsf_path) as src_wsf,
        rasterio.open(cell_ids_path) as src_ids,
    ):
        if (src_pca.height, src_pca.width) != (src_wsf.height, src_wsf.width):
            raise ValueError("PCA raster and WSF feature raster must have the same shape.")
        if (src_pca.height, src_pca.width) != (src_ids.height, src_ids.width):
            raise ValueError("PCA raster and cell-id raster must have the same shape.")
        if (
            src_pca.transform != src_wsf.transform
            or src_pca.transform != src_ids.transform
        ):
            raise ValueError("Transforms differ among input rasters.")
        if src_pca.crs != src_wsf.crs or src_pca.crs != src_ids.crs:
            raise ValueError("CRS differs among input rasters.")

        pca = src_pca.read().astype(np.float32, copy=False)
        wsf_feats = src_wsf.read().astype(np.float32, copy=False)
        cell_ids = src_ids.read(1).astype(np.int32, copy=False)
        profile = src_pca.profile.copy()

    if not (1 <= wsf_band_1based <= wsf_feats.shape[0]):
        raise ValueError(f"--wsf-band must be between 1 and {wsf_feats.shape[0]}")
    wsf_bin = wsf_feats[wsf_band_1based - 1]
    return pca, wsf_feats, wsf_bin, cell_ids, profile


def standardize_channels(x: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    """
    Standardize each channel over valid pixels only. Invalid pixels become zero.
    x shape: (C, H, W)
    valid_mask shape: (H, W), boolean
    """
    x_std = np.zeros_like(x, dtype=np.float32)
    stats: List[dict] = []
    for c in range(x.shape[0]):
        vals = x[c][valid_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean = 0.0
            std = 1.0
        else:
            mean = float(vals.mean())
            std = float(vals.std())
            if std < 1e-8:
                std = 1.0
        xc = x[c].copy()
        xc[~np.isfinite(xc)] = mean
        xc = (xc - mean) / std
        xc[~valid_mask] = 0.0
        x_std[c] = xc.astype(np.float32, copy=False)
        stats.append({"band": int(c + 1), "mean": mean, "std": std})
    return x_std, stats


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class DilatedResidualScoreNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int = 32,
        depth: int = 4,
        dilations: List[int] | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        dilation_schedule = dilations or [1, 2, 4, 8]
        if any(d < 1 for d in dilation_schedule):
            raise ValueError("all dilations must be positive integers")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(hidden, dilation_schedule[i % len(dilation_schedule)])
                for i in range(depth)
            ]
        )
        self.head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.out_act = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.blocks(y)
        return self.out_act(self.head(y))


def tv_loss_2d(y: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    y: (1, 1, H, W)
    valid_mask: (1, 1, H, W), float 0/1
    """
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])

    my = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    mx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]

    dy_num = (dy * my).sum()
    dx_num = (dx * mx).sum()
    denom = my.sum() + mx.sum() + 1e-8
    return (dy_num + dx_num) / denom


@torch.no_grad()
def feature_edge_weights(x: torch.Tensor, valid_mask: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise smoothing weights from feature gradients.

    x: (1, C, H, W), standardized input features.
    valid_mask: (1, 1, H, W), float 0/1.
    Returns weights for y-direction pairs and x-direction pairs.
    """
    grad_y = torch.sqrt(torch.mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2, dim=1, keepdim=True) + 1e-8)
    grad_x = torch.sqrt(torch.mean((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2, dim=1, keepdim=True) + 1e-8)
    wy = torch.exp(-float(alpha) * grad_y)
    wx = torch.exp(-float(alpha) * grad_x)
    wy = wy * valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    wx = wx * valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    return wy, wx


def edge_aware_tv_loss_2d(
    y: torch.Tensor,
    edge_weight_y: torch.Tensor,
    edge_weight_x: torch.Tensor,
) -> torch.Tensor:
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    denom = edge_weight_y.sum() + edge_weight_x.sum() + 1e-8
    return ((dy * edge_weight_y).sum() + (dx * edge_weight_x).sum()) / denom


def make_coarse_boundary_masks(cell_ids: np.ndarray, allocation_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid_y = allocation_mask[1:, :] & allocation_mask[:-1, :]
    valid_x = allocation_mask[:, 1:] & allocation_mask[:, :-1]
    boundary_y = valid_y & (cell_ids[1:, :] != cell_ids[:-1, :])
    boundary_x = valid_x & (cell_ids[:, 1:] != cell_ids[:, :-1])
    return boundary_y.astype(np.float32, copy=False), boundary_x.astype(np.float32, copy=False)


def coarse_boundary_loss_2d(
    y: torch.Tensor,
    boundary_mask_y: torch.Tensor,
    boundary_mask_x: torch.Tensor,
) -> torch.Tensor:
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    denom = boundary_mask_y.sum() + boundary_mask_x.sum() + 1e-8
    return ((dy * boundary_mask_y).sum() + (dx * boundary_mask_x).sum()) / denom


def build_cell_mapping(cell_ids: np.ndarray, lookup: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Returns:
    - valid_mask_flat: pixels with cell_id > 0 and coarse target present
    - valid_cell_ids_flat: cell ids for valid pixels
    - coarse_targets: float32 array, one per compact coarse index
    - cellid_to_compact: mapping original cell_id -> compact [0..K-1]
    """
    flat_ids = cell_ids.reshape(-1)
    keep = flat_ids > 0
    if not np.any(keep):
        raise ValueError("No valid cell IDs found in fine raster.")

    present_ids = np.unique(flat_ids[keep])
    present_ids = np.array([cid for cid in present_ids if int(cid) in lookup], dtype=np.int64)
    if present_ids.size == 0:
        raise ValueError("No coarse cell IDs in raster matched IDs in lookup CSV.")

    cellid_to_compact = {int(cid): i for i, cid in enumerate(present_ids.tolist())}
    valid_mask_flat = np.array([(int(cid) in cellid_to_compact) for cid in flat_ids], dtype=bool)
    valid_cell_ids_flat = flat_ids[valid_mask_flat].astype(np.int64, copy=False)
    coarse_targets = np.array([lookup[int(cid)] for cid in present_ids], dtype=np.float32)
    return valid_mask_flat, valid_cell_ids_flat, coarse_targets, cellid_to_compact


def aggregate_to_cells(pred_flat_valid: torch.Tensor, compact_idx_valid: torch.Tensor, n_cells: int) -> torch.Tensor:
    coarse = torch.zeros(n_cells, device=pred_flat_valid.device, dtype=pred_flat_valid.dtype)
    coarse.index_add_(0, compact_idx_valid, pred_flat_valid)
    return coarse


def normalize_scores_by_cell(
    score: torch.Tensor,
    valid_linear_idx: torch.Tensor,
    compact_idx_valid: torch.Tensor,
    coarse_targets: torch.Tensor,
    n_cells: int,
    score_floor: float = 1e-6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert positive raw scores to mass-preserving fine predictions.

    score shape: (1, 1, H, W)
    valid_linear_idx: flattened pixel indices used for training/allocation
    compact_idx_valid: compact coarse-cell index for each valid pixel
    coarse_targets: target mass for each compact coarse-cell index
    """
    score_flat = score.reshape(-1)
    valid_scores = score_flat.index_select(0, valid_linear_idx).clamp_min(score_floor)
    cell_sums = aggregate_to_cells(valid_scores, compact_idx_valid, n_cells)
    target_valid = coarse_targets.index_select(0, compact_idx_valid)
    denom_valid = cell_sums.index_select(0, compact_idx_valid) + eps
    pred_valid = valid_scores / denom_valid * target_valid

    pred_flat = torch.zeros_like(score_flat)
    pred_flat = pred_flat.scatter(0, valid_linear_idx, pred_valid)
    return pred_flat.reshape_as(score)


def score_sum_diagnostics(
    score: torch.Tensor,
    valid_linear_idx: torch.Tensor,
    compact_idx_valid: torch.Tensor,
    represented_cell_idx: torch.Tensor,
    n_cells: int,
    score_floor: float = 1e-6,
    near_zero_threshold: float = 1e-6,
) -> Dict[str, float]:
    score_flat = score.reshape(-1)
    valid_scores = score_flat.index_select(0, valid_linear_idx).clamp_min(score_floor)
    cell_sums = aggregate_to_cells(valid_scores, compact_idx_valid, n_cells)
    represented_sums = cell_sums.index_select(0, represented_cell_idx).detach().cpu().numpy()
    return {
        "min_cell_score_sum": float(np.min(represented_sums)),
        "median_cell_score_sum": float(np.median(represented_sums)),
        "max_cell_score_sum": float(np.max(represented_sums)),
        "n_near_zero_cell_sums": int(np.sum(represented_sums <= near_zero_threshold)),
    }


def renormalize_by_cell(
    raw_pred: np.ndarray,
    cell_ids: np.ndarray,
    lookup: Dict[int, float],
    value_column_name: str = "ghsl_value_adj",
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each coarse cell with target > 0, scale fine predictions so the cell sum matches target exactly.
    Cells with target <= 0 become zero.
    """
    out = np.zeros_like(raw_pred, dtype=np.float32)
    flat_pred = raw_pred.reshape(-1)
    flat_ids = cell_ids.reshape(-1)
    flat_valid = np.ones_like(flat_ids, dtype=bool) if valid_mask is None else valid_mask.reshape(-1).astype(bool)
    out_flat = out.reshape(-1)

    unique_ids = np.unique(flat_ids[(flat_ids > 0) & flat_valid])
    for cid in unique_ids:
        target = float(lookup.get(int(cid), 0.0))
        mask = (flat_ids == cid) & flat_valid
        if target <= 0:
            out_flat[mask] = 0.0
            continue
        vals = flat_pred[mask]
        s = float(vals.sum())
        if s <= 0:
            # uniform fallback within represented fine pixels of that coarse cell
            n = int(mask.sum())
            if n > 0:
                out_flat[mask] = target / n
            continue
        out_flat[mask] = vals * (target / s)
    return out


def compute_metrics(
    pred: np.ndarray,
    pred_norm: np.ndarray,
    cell_ids: np.ndarray,
    lookup: Dict[int, float],
    valid_mask: np.ndarray,
    mass_prefix: str = "mass_preservation",
) -> Dict[str, float]:
    unique_ids = np.unique(cell_ids[cell_ids > 0])
    targets = []
    raw_sums = []
    norm_sums = []
    for cid in unique_ids:
        if int(cid) not in lookup:
            continue
        mask = (cell_ids == cid) & valid_mask
        if not np.any(mask):
            continue
        target = float(lookup[int(cid)])
        targets.append(target)
        raw_sums.append(float(np.nansum(pred[mask])))
        norm_sums.append(float(np.nansum(pred_norm[mask])))
    t = np.array(targets, dtype=np.float64)
    r = np.array(raw_sums, dtype=np.float64)
    n = np.array(norm_sums, dtype=np.float64)

    raw_mae = float(np.mean(np.abs(r - t))) if t.size else np.nan
    raw_rmse = float(np.sqrt(np.mean((r - t) ** 2))) if t.size else np.nan
    mass_errors = n - t
    mass_mae = float(np.mean(np.abs(mass_errors))) if t.size else np.nan
    mass_rmse = float(np.sqrt(np.mean(mass_errors**2))) if t.size else np.nan
    mass_max_abs = float(np.max(np.abs(mass_errors))) if t.size else np.nan
    raw_total = float(r.sum())
    norm_total = float(n.sum())
    target_total = float(t.sum())
    mass_total_error = float(norm_total - target_total)

    return {
        "raw_score_coarse_mae": raw_mae,
        "raw_score_coarse_rmse": raw_rmse,
        "raw_score_total": raw_total,
        "target_total": target_total,
        "norm_total": norm_total,
        f"{mass_prefix}_mae": mass_mae,
        f"{mass_prefix}_rmse": mass_rmse,
        f"{mass_prefix}_max_abs_error": mass_max_abs,
        f"{mass_prefix}_total_error": mass_total_error,
        f"{mass_prefix}_cells_checked": int(t.size),
    }


def scaled_raw_coarse_mse(
    pred: np.ndarray,
    cell_ids: np.ndarray,
    lookup: Dict[int, float],
    valid_mask: np.ndarray,
) -> float:
    unique_ids = np.unique(cell_ids[cell_ids > 0])
    targets = []
    raw_sums = []
    for cid in unique_ids:
        if int(cid) not in lookup:
            continue
        mask = (cell_ids == cid) & valid_mask
        if not np.any(mask):
            continue
        targets.append(float(lookup[int(cid)]))
        raw_sums.append(float(np.nansum(pred[mask])))
    t = np.array(targets, dtype=np.float64)
    r = np.array(raw_sums, dtype=np.float64)
    if not t.size:
        return np.nan
    scale = float(t.mean()) + 1e-6
    return float(np.mean(((r / scale) - (t / scale)) ** 2))


def write_raster(out_path: Path, arr: np.ndarray, profile: dict) -> None:
    profile = profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32, copy=False), 1)


def plot_losses(history: List[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    total = [h["loss"] for h in history]
    raw_coarse = [h["raw_coarse_loss"] for h in history]
    tv = [h["tv_loss"] for h in history]
    edge_tv = [h["edge_tv_loss"] for h in history]
    boundary = [h["boundary_loss"] for h in history]
    wsf = [h["wsf_loss"] for h in history]

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, total, label="total loss")
    plt.plot(epochs, raw_coarse, label="raw coarse loss")
    plt.plot(epochs, tv, label="TV loss")
    plt.plot(epochs, edge_tv, label="edge-aware TV loss")
    plt.plot(epochs, boundary, label="coarse boundary loss")
    plt.plot(epochs, wsf, label="WSF loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Dilated residual embeddings + WSF diffnorm training")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.score_floor <= 0:
        raise ValueError("--score-floor must be positive")
    seed_everything(args.seed)
    device = choose_device()
    print(f"[INFO] Using device: {device}")

    pca_path = Path(args.pca).expanduser().resolve()
    wsf_path = Path(args.wsf_features).expanduser().resolve()
    cell_ids_path = Path(args.cell_ids).expanduser().resolve()
    lookup_path = Path(args.lookup).expanduser().resolve()

    lookup = read_lookup(lookup_path, args.value_column)
    pca, wsf_feats, wsf_bin, cell_ids, profile = read_rasters(
        pca_path,
        wsf_path,
        cell_ids_path,
        args.wsf_band,
    )

    c_pca, h, w = pca.shape
    c_wsf = wsf_feats.shape[0]
    print(f"[INFO] PCA raster shape: bands={c_pca}, height={h}, width={w}")
    print(f"[INFO] WSF feature raster shape: bands={c_wsf}, height={h}, width={w}")

    valid_mask = (cell_ids > 0) & np.all(np.isfinite(pca), axis=0)
    valid_mask &= np.isfinite(wsf_bin)
    valid_pixels = int(valid_mask.sum())
    print(f"[INFO] Valid fine pixels: {valid_pixels:,}")

    x_full = np.concatenate([pca, wsf_feats], axis=0)
    x_std, channel_stats = standardize_channels(x_full, valid_mask)

    # binary WSF support from selected band
    wsf_support = np.where(np.isfinite(wsf_bin) & (wsf_bin > 0), 1.0, 0.0).astype(np.float32)
    wsf_support[~valid_mask] = 0.0

    cell_lookup_mask_flat, _, coarse_targets_np, cellid_to_compact = build_cell_mapping(cell_ids, lookup)
    train_valid_flat = valid_mask.reshape(-1) & cell_lookup_mask_flat
    if not np.any(train_valid_flat):
        raise ValueError("No fine pixels are both feature-valid and matched to lookup coarse cells.")
    train_valid_cell_ids_flat = cell_ids.reshape(-1)[train_valid_flat].astype(np.int64, copy=False)
    compact_idx_np = np.array([cellid_to_compact[int(cid)] for cid in train_valid_cell_ids_flat], dtype=np.int64)
    valid_linear_idx_np = np.where(train_valid_flat)[0].astype(np.int64, copy=False)
    represented_cell_idx_np = np.unique(compact_idx_np).astype(np.int64, copy=False)
    allocation_mask = train_valid_flat.reshape(h, w)
    n_coarse = int(coarse_targets_np.shape[0])
    print(f"[INFO] Number of coarse cells: {n_coarse}")
    print(f"[INFO] Coarse cells represented by allocation pixels: {represented_cell_idx_np.size}")
    print(f"[INFO] Fine pixels used for allocation: {valid_linear_idx_np.size:,}")
    print(f"[INFO] Total target mass ({args.value_column}): {coarse_targets_np.sum():.4f}")

    # tensors
    x_t = torch.from_numpy(x_std[None, ...]).to(device=device, dtype=torch.float32)
    valid_t = torch.from_numpy(allocation_mask.astype(np.float32)[None, None, ...]).to(device=device, dtype=torch.float32)
    wsf_t = torch.from_numpy(wsf_support[None, None, ...]).to(device=device, dtype=torch.float32)
    edge_weight_y_t, edge_weight_x_t = feature_edge_weights(x_t, valid_t, args.edge_tv_alpha)
    boundary_mask_y_np, boundary_mask_x_np = make_coarse_boundary_masks(cell_ids, allocation_mask)
    boundary_mask_y_t = torch.from_numpy(boundary_mask_y_np[None, None, ...]).to(device=device, dtype=torch.float32)
    boundary_mask_x_t = torch.from_numpy(boundary_mask_x_np[None, None, ...]).to(device=device, dtype=torch.float32)
    valid_linear_idx_t = torch.from_numpy(valid_linear_idx_np).to(device=device, dtype=torch.long)
    compact_idx_t = torch.from_numpy(compact_idx_np).to(device=device, dtype=torch.long)
    represented_cell_idx_t = torch.from_numpy(represented_cell_idx_np).to(device=device, dtype=torch.long)
    coarse_targets_t = torch.from_numpy(coarse_targets_np).to(device=device, dtype=torch.float32)

    model = DilatedResidualScoreNet(
        in_channels=x_std.shape[0],
        hidden=args.hidden,
        depth=args.depth,
        dilations=args.dilations,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: List[dict] = []
    near_zero_threshold = 1e-6

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        score = model(x_t) * valid_t
        pred_mass = normalize_scores_by_cell(
            score=score,
            valid_linear_idx=valid_linear_idx_t,
            compact_idx_valid=compact_idx_t,
            coarse_targets=coarse_targets_t,
            n_cells=n_coarse,
            score_floor=args.score_floor,
        )
        denom_diag = score_sum_diagnostics(
            score=score,
            valid_linear_idx=valid_linear_idx_t,
            compact_idx_valid=compact_idx_t,
            represented_cell_idx=represented_cell_idx_t,
            n_cells=n_coarse,
            score_floor=args.score_floor,
            near_zero_threshold=near_zero_threshold,
        )
        score_flat_valid = score.reshape(-1).index_select(0, valid_linear_idx_t)
        raw_coarse_score = aggregate_to_cells(score_flat_valid, compact_idx_t, n_coarse)
        raw_coarse_scale = coarse_targets_t.mean() + 1e-6
        raw_coarse_loss = F.mse_loss(raw_coarse_score / raw_coarse_scale, coarse_targets_t / raw_coarse_scale)

        pred_mass_flat_valid = pred_mass.reshape(-1).index_select(0, valid_linear_idx_t)
        coarse_mass = aggregate_to_cells(pred_mass_flat_valid, compact_idx_t, n_coarse)
        coarse_mass_error = coarse_mass.index_select(0, represented_cell_idx_t) - coarse_targets_t.index_select(
            0, represented_cell_idx_t
        )
        mass_preservation_rmse = torch.sqrt(torch.mean(coarse_mass_error**2))

        tv = tv_loss_2d(pred_mass, valid_t)
        edge_tv = edge_aware_tv_loss_2d(pred_mass, edge_weight_y_t, edge_weight_x_t)
        boundary_loss = coarse_boundary_loss_2d(pred_mass, boundary_mask_y_t, boundary_mask_x_t)

        total_mass = pred_mass.sum() + 1e-8
        outside_mass = (pred_mass * (1.0 - wsf_t)).sum()
        wsf_loss = outside_mass / total_mass

        loss = (
            args.raw_coarse_weight * raw_coarse_loss
            + args.tv_weight * tv
            + args.edge_tv_weight * edge_tv
            + args.boundary_weight * boundary_loss
            + args.wsf_weight * wsf_loss
        )
        loss.backward()
        optimizer.step()

        record = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu().item()),
            "raw_coarse_loss": float(raw_coarse_loss.detach().cpu().item()),
            "tv_loss": float(tv.detach().cpu().item()),
            "edge_tv_loss": float(edge_tv.detach().cpu().item()),
            "boundary_loss": float(boundary_loss.detach().cpu().item()),
            "wsf_loss": float(wsf_loss.detach().cpu().item()),
            "mass_preservation_rmse": float(mass_preservation_rmse.detach().cpu().item()),
            **denom_diag,
        }
        history.append(record)

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[INFO] Epoch {epoch:04d} | "
                f"loss={record['loss']:.4f} | "
                f"raw_coarse={record['raw_coarse_loss']:.4f} | "
                f"tv={record['tv_loss']:.4f} | "
                f"edge_tv={record['edge_tv_loss']:.4f} | "
                f"boundary={record['boundary_loss']:.4f} | "
                f"wsf={record['wsf_loss']:.6f} | "
                f"mass_rmse={record['mass_preservation_rmse']:.6g} | "
                f"min_score_sum={record['min_cell_score_sum']:.6g} | "
                f"near_zero_sums={record['n_near_zero_cell_sums']}"
            )
            if record["n_near_zero_cell_sums"] > 0:
                print(
                    "[WARN] Differentiable normalization has near-zero cell score sums; "
                    "consider increasing --score-floor."
                )

    # inference
    model.eval()
    with torch.no_grad():
        score = model(x_t) * valid_t
        pred_mass = normalize_scores_by_cell(
            score=score,
            valid_linear_idx=valid_linear_idx_t,
            compact_idx_valid=compact_idx_t,
            coarse_targets=coarse_targets_t,
            n_cells=n_coarse,
            score_floor=args.score_floor,
        )
        final_denominator_diagnostics = score_sum_diagnostics(
            score=score,
            valid_linear_idx=valid_linear_idx_t,
            compact_idx_valid=compact_idx_t,
            represented_cell_idx=represented_cell_idx_t,
            n_cells=n_coarse,
            score_floor=args.score_floor,
            near_zero_threshold=near_zero_threshold,
        )
    pred_np = score.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
    pred_np[~allocation_mask] = np.nan

    pred_mass_np = pred_mass.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
    pred_mass_np[~allocation_mask] = np.nan

    pred_norm_np = renormalize_by_cell(
        raw_pred=np.maximum(np.nan_to_num(pred_np, nan=0.0), args.score_floor),
        cell_ids=cell_ids,
        lookup=lookup,
        value_column_name=args.value_column,
        valid_mask=allocation_mask,
    ).astype(np.float32, copy=False)
    pred_norm_np[~allocation_mask] = np.nan

    differentiable_metrics = compute_metrics(
        pred=np.nan_to_num(pred_np, nan=0.0),
        pred_norm=pred_mass_np.copy(),
        cell_ids=cell_ids,
        lookup=lookup,
        valid_mask=allocation_mask,
        mass_prefix="differentiable_mass_preservation",
    )
    final_metrics = compute_metrics(
        pred=np.nan_to_num(pred_np, nan=0.0),
        pred_norm=pred_norm_np.copy(),
        cell_ids=cell_ids,
        lookup=lookup,
        valid_mask=allocation_mask,
        mass_prefix="final_mass_preservation",
    )
    metrics = {**differentiable_metrics, **final_metrics}
    metrics["mass_preservation_mae"] = metrics["final_mass_preservation_mae"]
    metrics["mass_preservation_rmse"] = metrics["final_mass_preservation_rmse"]
    metrics["mass_preservation_max_abs_error"] = metrics["final_mass_preservation_max_abs_error"]
    metrics["mass_preservation_total_error"] = metrics["final_mass_preservation_total_error"]
    metrics["mass_preservation_cells_checked"] = metrics["final_mass_preservation_cells_checked"]
    metrics.update(final_denominator_diagnostics)
    metrics["final_raw_coarse_scaled_mse"] = scaled_raw_coarse_mse(
        pred=np.nan_to_num(pred_np, nan=0.0),
        cell_ids=cell_ids,
        lookup=lookup,
        valid_mask=allocation_mask,
    )
    with torch.no_grad():
        final_tv = tv_loss_2d(pred_mass, valid_t)
        final_edge_tv = edge_aware_tv_loss_2d(pred_mass, edge_weight_y_t, edge_weight_x_t)
        final_boundary = coarse_boundary_loss_2d(pred_mass, boundary_mask_y_t, boundary_mask_x_t)
        final_total_mass = pred_mass.sum() + 1e-8
        final_outside_mass = (pred_mass * (1.0 - wsf_t)).sum()
        final_wsf_loss = final_outside_mass / final_total_mass
    metrics["final_tv_loss"] = float(final_tv.detach().cpu().item())
    metrics["final_edge_tv_loss"] = float(final_edge_tv.detach().cpu().item())
    metrics["final_boundary_loss"] = float(final_boundary.detach().cpu().item())
    metrics["final_wsf_loss"] = float(final_wsf_loss.detach().cpu().item())

    represented_targets_np = coarse_targets_np[represented_cell_idx_np]
    max_target = float(np.max(represented_targets_np)) if represented_targets_np.size else 0.0
    target_total = float(metrics["target_total"])
    final_total_error = abs(float(metrics["final_mass_preservation_total_error"]))
    final_max_abs_error = float(metrics["final_mass_preservation_max_abs_error"])
    if target_total > 0 and final_total_error / target_total > 1e-6:
        print(
            "[WARN] Final normalized raster failed total mass-preservation tolerance: "
            f"abs(total_error) / target_total = {final_total_error / target_total:.6g}"
        )
    if max_target > 0 and final_max_abs_error > 1e-3 * max_target:
        print(
            "[WARN] Final normalized raster failed per-cell mass-preservation tolerance: "
            f"max_abs_error = {final_max_abs_error:.6g}, threshold = {1e-3 * max_target:.6g}"
        )
    if metrics["n_near_zero_cell_sums"] > 0:
        print(
            "[WARN] Differentiable normalization final pass has near-zero cell score sums; "
            "the exported raster was CPU-renormalized exactly, but training may be numerically unstable."
        )

    # additional WSF diagnostics
    raw_total_mass = float(np.nansum(pred_np))
    norm_total_mass = float(np.nansum(pred_norm_np))
    raw_inside_wsf = float(np.nansum(np.where(wsf_support > 0, np.nan_to_num(pred_np, nan=0.0), 0.0)))
    norm_inside_wsf = float(np.nansum(np.where(wsf_support > 0, np.nan_to_num(pred_norm_np, nan=0.0), 0.0)))
    raw_inside_frac = raw_inside_wsf / raw_total_mass if raw_total_mass > 0 else np.nan
    norm_inside_frac = norm_inside_wsf / norm_total_mass if norm_total_mass > 0 else np.nan
    metrics["raw_score_mass_fraction_inside_wsf"] = float(raw_inside_frac)
    metrics["norm_mass_fraction_inside_wsf"] = float(norm_inside_frac)
    metrics["norm_mass_fraction_outside_wsf"] = float(1.0 - norm_inside_frac) if np.isfinite(norm_inside_frac) else np.nan

    pred_out = Path(args.pred_out).expanduser().resolve()
    pred_norm_out = Path(args.pred_norm_out).expanduser().resolve()
    report_out = Path(args.report).expanduser().resolve()

    write_raster(pred_out, pred_np, profile)
    print(f"[INFO] Saved raw dilated CNN score raster to: {pred_out}")

    write_raster(pred_norm_out, pred_norm_np, profile)
    print(f"[INFO] Saved mass-preserving normalized prediction raster to: {pred_norm_out}")

    if args.model_out:
        model_out = Path(args.model_out).expanduser().resolve()
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_out)
        print(f"[INFO] Saved model weights to: {model_out}")

    if args.loss_plot:
        loss_plot = Path(args.loss_plot).expanduser().resolve()
        plot_losses(history, loss_plot)
        print(f"[INFO] Saved loss plot to: {loss_plot}")

    report = {
        "device": str(device),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "tv_weight": float(args.tv_weight),
        "edge_tv_weight": float(args.edge_tv_weight),
        "edge_tv_alpha": float(args.edge_tv_alpha),
        "boundary_weight": float(args.boundary_weight),
        "wsf_weight": float(args.wsf_weight),
        "raw_coarse_weight": float(args.raw_coarse_weight),
        "score_floor": float(args.score_floor),
        "near_zero_score_sum_threshold": float(near_zero_threshold),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "dilations": [int(d) for d in args.dilations],
        "value_column": args.value_column,
        "wsf_band": int(args.wsf_band),
        "valid_fine_pixels": int(valid_pixels),
        "allocation_fine_pixels": int(valid_linear_idx_np.size),
        "n_coarse_cells": int(n_coarse),
        "n_coarse_cells_represented_by_allocation_pixels": int(represented_cell_idx_np.size),
        "n_pca_bands": int(c_pca),
        "n_wsf_feature_bands": int(c_wsf),
        "n_coarse_boundary_y_pairs": int(boundary_mask_y_np.sum()),
        "n_coarse_boundary_x_pairs": int(boundary_mask_x_np.sum()),
        "channel_stats": channel_stats,
        "output_descriptions": {
            "pred_out": "raw positive dilated CNN score surface for debugging; not mass-preserving",
            "pred_norm_out": "exact CPU-renormalized mass-preserving fine prediction",
        },
        "metric_groups": {
            "raw_score_diagnostics": [
                "raw_score_coarse_mae",
                "raw_score_coarse_rmse",
                "raw_score_total",
                "raw_score_mass_fraction_inside_wsf",
                "final_raw_coarse_scaled_mse",
            ],
            "denominator_diagnostics": [
                "min_cell_score_sum",
                "median_cell_score_sum",
                "max_cell_score_sum",
                "n_near_zero_cell_sums",
            ],
            "differentiable_mass_preservation_checks": [
                "differentiable_mass_preservation_mae",
                "differentiable_mass_preservation_rmse",
                "differentiable_mass_preservation_max_abs_error",
                "differentiable_mass_preservation_total_error",
                "differentiable_mass_preservation_cells_checked",
            ],
            "final_mass_preservation_checks": [
                "final_mass_preservation_mae",
                "final_mass_preservation_rmse",
                "final_mass_preservation_max_abs_error",
                "final_mass_preservation_total_error",
                "final_mass_preservation_cells_checked",
            ],
            "mass_preservation_checks": [
                "mass_preservation_mae",
                "mass_preservation_rmse",
                "mass_preservation_max_abs_error",
                "mass_preservation_total_error",
                "mass_preservation_cells_checked",
            ],
            "spatial_plausibility_diagnostics": [
                "norm_mass_fraction_inside_wsf",
                "norm_mass_fraction_outside_wsf",
            ],
            "regularization_losses": [
                "final_tv_loss",
                "final_edge_tv_loss",
                "final_boundary_loss",
                "final_wsf_loss",
            ],
        },
        "metrics_note": (
            "Differentiable mass-preservation fields describe the training-time PyTorch normalization. "
            "Final mass-preservation fields describe the CPU-renormalized raster written to pred_norm_out. "
            "These are sanity checks from enforced within-cell normalization, not validation metrics for predictive skill."
        ),
        "metrics": metrics,
        "history_tail": history[-10:],
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with report_out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved report to: {report_out}")

    print("[INFO] Final metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:,.6f}")


if __name__ == "__main__":
    main()
