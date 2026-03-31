#!/usr/bin/env python3
"""
11_train_embed_only.py

Baseline 1: embeddings-only downscaling model.

Trains a small fully convolutional network on the full fine grid using only
PCA-reduced Google embeddings as input. The model predicts a positive fine-scale
surface, which is aggregated to coarse GHSL cell IDs for supervision. After
training, predictions are renormalized within each coarse cell to exactly match
coarse GHSL totals.

Inputs
------
--pca         Fine-grid PCA raster (bands x H x W), e.g. mosaic_accra_2019_pca8.tif
--cell-ids    Fine-grid raster of GHSL cell IDs, e.g. cropped_ghsl_cell_ids.tif
--lookup      Lookup CSV from step 3; must contain cell_id and target value column
--value-column  Column in lookup CSV, e.g. ghsl_value_adj

Outputs
-------
--pred-out       Raw model prediction raster (positive fine surface)
--pred-norm-out  Renormalized prediction raster (exact coarse-cell mass preserved)
--report         JSON report with losses and diagnostics
--loss-plot      Optional PNG loss curve
--model-out      Saved PyTorch model state dict

Example
-------
python 11_train_embed_only.py \
  --pca ~/data/aef_accra_2019/mosaic_accra_2019_pca8.tif \
  --cell-ids ~/data/GHSL_BUILD/cropped_ghsl_cell_ids.tif \
  --lookup ~/data/GHSL_BUILD/cropped_ghsl_cell_lookup.csv \
  --value-column ghsl_value_adj \
  --pred-out ~/data/outputs/embed_only_raw.tif \
  --pred-norm-out ~/data/outputs/embed_only_norm.tif \
  --report ~/data/outputs/embed_only_report.json \
  --loss-plot ~/data/outputs/embed_only_loss.png \
  --model-out ~/data/outputs/embed_only_model.pt \
  --epochs 500 \
  --lr 1e-3 \
  --tv-weight 1e-5 \
  --hidden 32 \
  --depth 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallConvNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        layers: List[nn.Module] = []
        c_in = in_channels
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(c_in, hidden, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = hidden
        layers.append(nn.Conv2d(c_in, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # strictly positive predictions
        return self.softplus(self.net(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train embeddings-only baseline on full fine grid")
    p.add_argument("--pca", required=True, help="Path to PCA raster")
    p.add_argument("--cell-ids", required=True, help="Path to fine-grid cell ID raster")
    p.add_argument("--lookup", required=True, help="Path to coarse-cell lookup CSV")
    p.add_argument("--value-column", default="ghsl_value_adj", help="Lookup column used as coarse target")
    p.add_argument("--pred-out", required=True, help="Output raw prediction raster")
    p.add_argument("--pred-norm-out", required=True, help="Output renormalized prediction raster")
    p.add_argument("--report", required=True, help="Output JSON report")
    p.add_argument("--loss-plot", default=None, help="Optional PNG path for loss curve")
    p.add_argument("--model-out", default=None, help="Optional path to save model state dict")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--tv-weight", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_raster(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32, copy=False)
        profile = src.profile.copy()
    return arr, profile


def load_lookup(path: Path, value_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cell_id", value_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lookup missing required columns: {sorted(missing)}")
    return df[["cell_id", value_column]].copy()


def prepare_inputs(
    pca_arr: np.ndarray,
    cell_ids_arr: np.ndarray,
    lookup_df: pd.DataFrame,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if cell_ids_arr.ndim == 3:
        if cell_ids_arr.shape[0] != 1:
            raise ValueError("cell ID raster must have exactly one band")
        cell_ids_arr = cell_ids_arr[0]

    bands, height, width = pca_arr.shape
    if cell_ids_arr.shape != (height, width):
        raise ValueError("PCA raster and cell ID raster shapes do not match")

    flat_x = np.moveaxis(pca_arr, 0, -1).reshape(-1, bands)
    flat_ids = cell_ids_arr.reshape(-1).astype(np.int64, copy=False)

    valid_features = np.all(np.isfinite(flat_x), axis=1)
    valid_cells = flat_ids > 0
    valid_mask = valid_features & valid_cells

    # standardize using valid pixels only
    means = flat_x[valid_mask].mean(axis=0)
    stds = flat_x[valid_mask].std(axis=0)
    stds[stds == 0] = 1.0
    flat_x_std = flat_x.copy()
    flat_x_std[valid_mask] = (flat_x_std[valid_mask] - means) / stds
    flat_x_std[~valid_mask] = 0.0

    # map cell IDs to dense 0..n-1 indices, aligned with lookup
    lookup_df = lookup_df.sort_values("cell_id").reset_index(drop=True)
    cell_ids_lookup = lookup_df["cell_id"].to_numpy(dtype=np.int64)
    target_values = lookup_df.iloc[:, 1].to_numpy(dtype=np.float32)

    cell_id_to_dense = {cid: i for i, cid in enumerate(cell_ids_lookup)}
    dense_ids = np.full(flat_ids.shape, -1, dtype=np.int64)
    present = valid_mask.copy()
    for cid, dense_idx in cell_id_to_dense.items():
        dense_ids[flat_ids == cid] = dense_idx

    valid_mask &= dense_ids >= 0

    x_img = torch.from_numpy(np.moveaxis(flat_x_std.reshape(height, width, bands), -1, 0)).unsqueeze(0).to(device)
    valid_mask_t = torch.from_numpy(valid_mask.reshape(height, width)).to(device)
    dense_ids_t = torch.from_numpy(dense_ids).to(device)
    target_t = torch.from_numpy(target_values).to(device)
    means_t = torch.from_numpy(means.astype(np.float32))
    stds_t = torch.from_numpy(stds.astype(np.float32))

    return {
        "x_img": x_img,
        "valid_mask": valid_mask_t,
        "dense_ids": dense_ids_t,
        "target": target_t,
        "means": means_t,
        "stds": stds_t,
        "height": torch.tensor(height),
        "width": torch.tensor(width),
        "lookup_cell_ids": torch.from_numpy(cell_ids_lookup),
    }


def aggregate_by_cell(pred_flat: torch.Tensor, dense_ids_flat: torch.Tensor, n_cells: int) -> torch.Tensor:
    out = torch.zeros(n_cells, dtype=pred_flat.dtype, device=pred_flat.device)
    out.scatter_add_(0, dense_ids_flat, pred_flat)
    return out


def total_variation_loss(img: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    # img: [1, 1, H, W], valid_mask: [H, W]
    dx = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dy = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])

    vmx = (valid_mask[1:, :] & valid_mask[:-1, :]).unsqueeze(0).unsqueeze(0)
    vmy = (valid_mask[:, 1:] & valid_mask[:, :-1]).unsqueeze(0).unsqueeze(0)

    dx_mean = dx[vmx].mean() if vmx.any() else torch.tensor(0.0, device=img.device)
    dy_mean = dy[vmy].mean() if vmy.any() else torch.tensor(0.0, device=img.device)
    return dx_mean + dy_mean


@torch.no_grad()
def renormalize_by_cell(
    pred_img: torch.Tensor,
    valid_mask: torch.Tensor,
    dense_ids_flat: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # pred_img: [H, W]
    pred_flat = pred_img.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    pred_valid = pred_flat[valid_flat]
    ids_valid = dense_ids_flat[valid_flat]

    cell_sum = aggregate_by_cell(pred_valid, ids_valid, target.numel())
    scale = torch.zeros_like(target)
    pos = cell_sum > 0
    scale[pos] = target[pos] / cell_sum[pos]

    scaled_valid = pred_valid * scale[ids_valid]
    out = torch.zeros_like(pred_flat)
    out[valid_flat] = scaled_valid
    return out.reshape_as(pred_img)


def train_model(
    model: nn.Module,
    x_img: torch.Tensor,
    valid_mask: torch.Tensor,
    dense_ids_flat: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    tv_weight: float,
    log_every: int,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []

    valid_flat = valid_mask.reshape(-1)
    ids_valid = dense_ids_flat[valid_flat]

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        pred = model(x_img).squeeze(0).squeeze(0)  # [H, W]
        pred_valid = pred.reshape(-1)[valid_flat]
        coarse_pred = aggregate_by_cell(pred_valid, ids_valid, target.numel())

        coarse_loss = F.huber_loss(coarse_pred, target, reduction="mean", delta=100.0)
        tv_loss = total_variation_loss(pred.unsqueeze(0).unsqueeze(0), valid_mask)
        loss = coarse_loss + tv_weight * tv_loss

        loss.backward()
        opt.step()

        entry = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "coarse_loss": float(coarse_loss.item()),
            "tv_loss": float(tv_loss.item()),
        }
        history.append(entry)

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"[INFO] Epoch {epoch:04d} | "
                f"loss={entry['loss']:.4f} | "
                f"coarse={entry['coarse_loss']:.4f} | "
                f"tv={entry['tv_loss']:.4f}"
            )

    return model, history


@torch.no_grad()
def evaluate_predictions(
    raw_pred: torch.Tensor,
    norm_pred: torch.Tensor,
    valid_mask: torch.Tensor,
    dense_ids_flat: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    valid_flat = valid_mask.reshape(-1)
    ids_valid = dense_ids_flat[valid_flat]

    raw_valid = raw_pred.reshape(-1)[valid_flat]
    raw_coarse = aggregate_by_cell(raw_valid, ids_valid, target.numel())

    norm_valid = norm_pred.reshape(-1)[valid_flat]
    norm_coarse = aggregate_by_cell(norm_valid, ids_valid, target.numel())

    raw_mae = torch.mean(torch.abs(raw_coarse - target)).item()
    raw_rmse = torch.sqrt(torch.mean((raw_coarse - target) ** 2)).item()
    norm_mae = torch.mean(torch.abs(norm_coarse - target)).item()
    norm_rmse = torch.sqrt(torch.mean((norm_coarse - target) ** 2)).item()

    return {
        "raw_coarse_mae": float(raw_mae),
        "raw_coarse_rmse": float(raw_rmse),
        "norm_coarse_mae": float(norm_mae),
        "norm_coarse_rmse": float(norm_rmse),
        "target_total": float(target.sum().item()),
        "raw_total": float(raw_valid.sum().item()),
        "norm_total": float(norm_valid.sum().item()),
    }


def save_raster(path: Path, arr: np.ndarray, template_profile: dict) -> None:
    profile = template_profile.copy()
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32, copy=False), 1)


def plot_loss(history: List[Dict[str, float]], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    coarse = [h["coarse_loss"] for h in history]
    tv = [h["tv_loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="total loss")
    plt.plot(epochs, coarse, label="coarse loss")
    plt.plot(epochs, tv, label="TV loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Embeddings-only baseline training")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")

    pca_path = Path(args.pca).expanduser().resolve()
    cell_ids_path = Path(args.cell_ids).expanduser().resolve()
    lookup_path = Path(args.lookup).expanduser().resolve()

    pca_arr, pca_profile = load_raster(pca_path)
    cell_ids_arr, _ = load_raster(cell_ids_path)
    lookup_df = load_lookup(lookup_path, args.value_column)

    data = prepare_inputs(pca_arr, cell_ids_arr, lookup_df, device=device)
    x_img = data["x_img"]
    valid_mask = data["valid_mask"]
    dense_ids = data["dense_ids"]
    target = data["target"]

    print(f"[INFO] PCA raster shape: bands={x_img.shape[1]}, height={x_img.shape[2]}, width={x_img.shape[3]}")
    print(f"[INFO] Valid fine pixels: {int(valid_mask.sum().item()):,}")
    print(f"[INFO] Number of coarse cells: {target.numel():,}")
    print(f"[INFO] Total target mass ({args.value_column}): {target.sum().item():,.4f}")

    model = SmallConvNet(
        in_channels=x_img.shape[1],
        hidden=args.hidden,
        depth=args.depth,
    ).to(device)

    model, history = train_model(
        model=model,
        x_img=x_img,
        valid_mask=valid_mask,
        dense_ids_flat=dense_ids,
        target=target,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tv_weight=args.tv_weight,
        log_every=args.log_every,
    )

    model.eval()
    with torch.no_grad():
        raw_pred = model(x_img).squeeze(0).squeeze(0)
        norm_pred = renormalize_by_cell(raw_pred, valid_mask, dense_ids, target)

    metrics = evaluate_predictions(raw_pred, norm_pred, valid_mask, dense_ids, target)

    raw_np = raw_pred.detach().cpu().numpy().astype(np.float32, copy=False)
    norm_np = norm_pred.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_np = valid_mask.detach().cpu().numpy()
    raw_np[~valid_np] = np.nan
    norm_np[~valid_np] = np.nan

    pred_out = Path(args.pred_out).expanduser().resolve()
    pred_norm_out = Path(args.pred_norm_out).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    save_raster(pred_out, raw_np, pca_profile)
    print(f"[INFO] Saved raw prediction raster to: {pred_out}")
    save_raster(pred_norm_out, norm_np, pca_profile)
    print(f"[INFO] Saved renormalized prediction raster to: {pred_norm_out}")

    if args.model_out:
        model_out = Path(args.model_out).expanduser().resolve()
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_out)
        print(f"[INFO] Saved model weights to: {model_out}")

    if args.loss_plot:
        loss_plot_path = Path(args.loss_plot).expanduser().resolve()
        plot_loss(history, loss_plot_path)
        print(f"[INFO] Saved loss plot to: {loss_plot_path}")

    report = {
        "device": str(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tv_weight": args.tv_weight,
        "hidden": args.hidden,
        "depth": args.depth,
        "value_column": args.value_column,
        "valid_fine_pixels": int(valid_mask.sum().item()),
        "n_coarse_cells": int(target.numel()),
        "metrics": metrics,
        "history_tail": history[-10:],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved report to: {report_path}")

    print("[INFO] Final metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:,.6f}")


if __name__ == "__main__":
    main()
