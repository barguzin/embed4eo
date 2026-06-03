#!/usr/bin/env python3
"""
Create separate poster-ready validation figures from existing metric CSVs.

This script intentionally does not perform raster processing and intentionally
does not create a combined validation panel. Each validation indicator is
written as a separate figure file so it can be placed independently in a poster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FORMATS = ["png", "svg", "pdf"]
DEFAULT_MODELS = ["wsf_uniform", "embed_only", "embed_wsf"]
DEFAULT_MODEL_LABELS = ["WSF uniform", "Embedding only", "Embedding + WSF"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export separate poster validation figures from existing validation metric CSVs."
    )
    p.add_argument("--ghsl-csv", required=True, help="GHSL 10 m proxy metrics CSV.")
    p.add_argument("--viirs-csv", required=True, help="VIIRS metrics CSV.")
    p.add_argument("--gaia-csv", required=True, help="GAIA metrics CSV.")
    p.add_argument("--esa-csv", required=True, help="ESA WorldCover metrics CSV.")
    p.add_argument("--output-dir", required=True, help="Output directory for figures and selected metrics CSV.")
    p.add_argument("--models", nargs="+", default=None, help="Ordered raw model names.")
    p.add_argument("--model-labels", nargs="+", default=None, help="Display labels, same length as --models.")
    p.add_argument("--formats", nargs="+", default=DEFAULT_FORMATS, help="Output formats. Default: png svg pdf.")
    p.add_argument("--dpi", type=int, default=300, help="Output DPI for raster formats. Default: 300.")
    p.add_argument("--esa-topk-frac", type=float, default=0.05, help="ESA top-k fraction. Default: 0.05.")
    p.add_argument(
        "--include-secondary-metrics",
        action="store_true",
        help="Add optional second indicator where useful.",
    )
    p.add_argument(
        "--include-leakage",
        action="store_true",
        help="Export optional leakage/sanity-check figures.",
    )
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metric CSV not found: {path}")
    return pd.read_csv(path)


def infer_models(dataframes: Sequence[pd.DataFrame]) -> List[str]:
    models: List[str] = []
    for df in dataframes:
        if "model" not in df.columns:
            continue
        for model in df["model"].dropna().astype(str):
            if model not in models:
                models.append(model)
    if not models:
        raise ValueError("Could not infer model order from CSVs; pass --models explicitly.")
    return models


def labels_for_models(models: List[str], model_labels: Optional[List[str]]) -> List[str]:
    if model_labels is None:
        if models == DEFAULT_MODELS:
            return DEFAULT_MODEL_LABELS
        return models
    if len(model_labels) != len(models):
        raise ValueError("--model-labels must have the same length as --models.")
    return model_labels


def clean_formats(formats: Iterable[str]) -> List[str]:
    out = []
    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        if not fmt_clean:
            continue
        if fmt_clean not in out:
            out.append(fmt_clean)
    if not out:
        raise ValueError("--formats must include at least one format.")
    return out


def numeric_series(series: pd.Series, column_name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column {column_name!r} has no numeric values after conversion.")
    return out


def selected_rows(
    df: pd.DataFrame,
    metric_group: str,
    metric: str,
    models: Sequence[str],
    topk_frac: Optional[float] = None,
) -> pd.DataFrame:
    required = {"model", "metric_group", "metric", "value"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    sub = df[(df["metric_group"] == metric_group) & (df["metric"] == metric)].copy()
    if topk_frac is not None:
        if "topk_frac" not in sub.columns:
            raise ValueError(f"Metric {metric!r} requires column 'topk_frac'.")
        topk_values = pd.to_numeric(sub["topk_frac"], errors="coerce")
        sub = sub[np.isclose(topk_values, topk_frac, rtol=0, atol=1e-9)]

    missing = [model for model in models if model not in set(sub["model"].astype(str))]
    if missing:
        extra = f" at topk_frac={topk_frac}" if topk_frac is not None else ""
        raise ValueError(f"Missing metric {metric_group}/{metric}{extra} for model(s): {missing}")

    sub["model"] = pd.Categorical(sub["model"].astype(str), categories=list(models), ordered=True)
    sub["value"] = numeric_series(sub["value"], "value")
    return sub.sort_values("model")


def metric_values(
    df: pd.DataFrame,
    metric_group: str,
    metric: str,
    models: Sequence[str],
    topk_frac: Optional[float] = None,
) -> List[float]:
    sub = selected_rows(df, metric_group, metric, models, topk_frac)
    values = []
    for model in models:
        vals = sub[sub["model"] == model]["value"].dropna().to_numpy()
        if vals.size == 0:
            raise ValueError(f"No numeric value for {metric_group}/{metric} and model {model}.")
        values.append(float(vals[0]))
    return values


def setup_bar_axis(ax, labels: Sequence[str], ylabel: str, title: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)


def save_figure(fig, out_dir: Path, stem: str, formats: Sequence[str], dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def percent_label(frac: float) -> str:
    pct = frac * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}%"
    return f"{pct:.1f}%"


def append_selected(
    selected: List[dict],
    validation_source: str,
    figure_stem: str,
    rows: pd.DataFrame,
    models: Sequence[str],
    labels: Sequence[str],
    notes: str,
) -> None:
    for model, label in zip(models, labels):
        model_rows = rows[rows["model"].astype(str) == model]
        for _, row in model_rows.iterrows():
            selected.append(
                {
                    "validation_source": validation_source,
                    "figure_stem": figure_stem,
                    "model": model,
                    "model_label": label,
                    "metric_group": row.get("metric_group", ""),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", np.nan),
                    "scale_m": row.get("scale_m", np.nan),
                    "scale_factor": row.get("scale_factor", np.nan),
                    "topk_frac": row.get("topk_frac", np.nan),
                    "notes": notes,
                }
            )


def make_ghsl_rank_agreement(
    ghsl: pd.DataFrame,
    out_dir: Path,
    models: Sequence[str],
    labels: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    selected: List[dict],
) -> None:
    stem = "poster_validation_ghsl_rank_agreement"
    sub = selected_rows(ghsl, "scale", "spearman", models)
    if "scale_m" in sub.columns and pd.to_numeric(sub["scale_m"], errors="coerce").notna().any():
        x_col = "scale_m"
        x_label = "Aggregation scale (m)"
    elif "scale_factor" in sub.columns:
        x_col = "scale_factor"
        x_label = "Aggregation factor"
    else:
        raise ValueError("GHSL CSV must include either scale_m or scale_factor.")

    sub[x_col] = numeric_series(sub[x_col], x_col)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model, label in zip(models, labels):
        g = sub[sub["model"].astype(str) == model].sort_values(x_col)
        ax.plot(g[x_col].to_numpy(), g["value"].to_numpy(), marker="o", linewidth=2.0, label=label)
    ax.set_title("GHSL 10 m proxy: rank agreement by scale")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Spearman rank correlation")
    ax.grid(axis="both", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    save_figure(fig, out_dir, stem, formats, dpi)

    append_selected(
        selected,
        "GHSL",
        stem,
        sub,
        models,
        labels,
        "Spearman rank agreement with external GHSL 10 m proxy at each aggregation scale.",
    )


def make_viirs_correlation(
    viirs: pd.DataFrame,
    out_dir: Path,
    models: Sequence[str],
    labels: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    include_secondary: bool,
    selected: List[dict],
) -> None:
    stem = "poster_validation_viirs_correlation"
    metrics = [("spearman_log1p", "Spearman")]
    if include_secondary:
        metrics.append(("pearson_log1p", "Pearson"))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(len(models), dtype=float)
    width = 0.68 / len(metrics)
    for i, (metric, label) in enumerate(metrics):
        vals = metric_values(viirs, "correlation", metric, models)
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=label)
        rows = selected_rows(viirs, "correlation", metric, models)
        append_selected(
            selected,
            "VIIRS",
            stem,
            rows,
            models,
            labels,
            "Association with nighttime-light intensity; Spearman is rank-based and effectively invariant to log1p.",
        )

    setup_bar_axis(
        ax,
        labels,
        "Correlation",
        "VIIRS 2019: association with nighttime lights",
        ylim=(-1, 1),
    )
    if len(metrics) > 1:
        ax.legend(frameon=False)
    save_figure(fig, out_dir, stem, formats, dpi)


def make_gaia_binary_agreement(
    gaia: pd.DataFrame,
    out_dir: Path,
    models: Sequence[str],
    labels: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    include_secondary: bool,
    selected: List[dict],
) -> None:
    stem = "poster_validation_gaia_binary_agreement"
    metrics = [("iou", "IoU")]
    if include_secondary:
        metrics.append(("f1", "F1"))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(len(models), dtype=float)
    width = 0.68 / len(metrics)
    for i, (metric, label) in enumerate(metrics):
        vals = metric_values(gaia, "prevalence_matched_binary", metric, models)
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=label)
        rows = selected_rows(gaia, "prevalence_matched_binary", metric, models)
        append_selected(
            selected,
            "GAIA",
            stem,
            rows,
            models,
            labels,
            "Prevalence-matched binary agreement with GAIA 2019 impervious extent.",
        )

    setup_bar_axis(
        ax,
        labels,
        "Score",
        "GAIA 2019 impervious: prevalence-matched agreement",
        ylim=(0, 1),
    )
    if len(metrics) > 1:
        ax.legend(frameon=False)
    save_figure(fig, out_dir, stem, formats, dpi)


def make_esa_topk_overlap(
    esa: pd.DataFrame,
    out_dir: Path,
    models: Sequence[str],
    labels: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    esa_topk_frac: float,
    selected: List[dict],
) -> None:
    stem = "poster_validation_esa_topk_overlap"
    vals = metric_values(esa, "topk", "topk_esa_built_share", models, topk_frac=esa_topk_frac)
    rows = selected_rows(esa, "topk", "topk_esa_built_share", models, topk_frac=esa_topk_frac)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(np.arange(len(models)), vals, width=0.65)
    setup_bar_axis(
        ax,
        labels,
        "Share of top predicted cells in ESA built-up",
        f"ESA WorldCover 2020: top-{percent_label(esa_topk_frac)} overlap with built-up class",
        ylim=(0, 1),
    )
    save_figure(fig, out_dir, stem, formats, dpi)
    append_selected(
        selected,
        "ESA",
        stem,
        rows,
        models,
        labels,
        "Share of highest predicted cells that fall in ESA WorldCover built-up class.",
    )


def make_leakage_figure(
    esa: pd.DataFrame,
    gaia: pd.DataFrame,
    out_dir: Path,
    models: Sequence[str],
    labels: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    selected: List[dict],
) -> None:
    try:
        rows = selected_rows(esa, "mass_share", "share_mass_in_hard_nonbuilt", models)
        vals = [float(rows[rows["model"].astype(str) == model]["value"].iloc[0]) for model in models]
        stem = "poster_validation_esa_hard_nonbuilt_leakage"
        title = "ESA WorldCover: predicted mass in hard non-built classes"
        ylabel = "Share of predicted mass"
        source = "ESA"
        notes = "Predicted mass share in hard non-built ESA classes."
    except ValueError:
        rows = selected_rows(gaia, "summary", "share_mass_outside_gaia_impervious", models)
        vals = [float(rows[rows["model"].astype(str) == model]["value"].iloc[0]) for model in models]
        stem = "poster_validation_gaia_outside_impervious_mass"
        title = "GAIA: predicted mass outside impervious extent"
        ylabel = "Share of predicted mass"
        source = "GAIA"
        notes = "Predicted mass share outside GAIA impervious extent."

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(np.arange(len(models)), vals, width=0.65)
    setup_bar_axis(ax, labels, ylabel, title, ylim=(0, max(0.01, max(vals) * 1.15)))
    save_figure(fig, out_dir, stem, formats, dpi)
    append_selected(selected, source, stem, rows, models, labels, notes)


def write_selected_metrics(selected: List[dict], out_dir: Path) -> None:
    columns = [
        "validation_source",
        "figure_stem",
        "model",
        "model_label",
        "metric_group",
        "metric",
        "value",
        "scale_m",
        "scale_factor",
        "topk_frac",
        "notes",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected, columns=columns).to_csv(out_dir / "poster_validation_selected_metrics.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    formats = clean_formats(args.formats)

    ghsl = read_csv(Path(args.ghsl_csv).expanduser().resolve())
    viirs = read_csv(Path(args.viirs_csv).expanduser().resolve())
    gaia = read_csv(Path(args.gaia_csv).expanduser().resolve())
    esa = read_csv(Path(args.esa_csv).expanduser().resolve())

    models = args.models if args.models is not None else infer_models([ghsl, viirs, gaia, esa])
    labels = labels_for_models(models, args.model_labels)

    selected: List[dict] = []
    make_ghsl_rank_agreement(ghsl, out_dir, models, labels, formats, args.dpi, selected)
    make_viirs_correlation(
        viirs,
        out_dir,
        models,
        labels,
        formats,
        args.dpi,
        args.include_secondary_metrics,
        selected,
    )
    make_gaia_binary_agreement(
        gaia,
        out_dir,
        models,
        labels,
        formats,
        args.dpi,
        args.include_secondary_metrics,
        selected,
    )
    make_esa_topk_overlap(esa, out_dir, models, labels, formats, args.dpi, args.esa_topk_frac, selected)
    if args.include_leakage:
        make_leakage_figure(esa, gaia, out_dir, models, labels, formats, args.dpi, selected)

    write_selected_metrics(selected, out_dir)
    print(f"[INFO] Saved poster validation figures and selected metrics to: {out_dir}")


if __name__ == "__main__":
    main()
