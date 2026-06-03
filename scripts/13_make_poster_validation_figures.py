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
ESA_LEAKAGE_HARD_CLASSES = [80, 90, 95, 70]
ESA_LEAKAGE_SOFT_CLASSES = [30, 40, 60]
ESA_LEAKAGE_CLASS_LABELS = {
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    70: "Snow/ice",
    30: "Grassland",
    40: "Cropland",
    60: "Bare/sparse",
}


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
        "--bar-tight-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tighter y-axis ranges for poster bar charts. Default: enabled.",
    )
    p.add_argument(
        "--bar-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw numeric value labels above poster bar charts. Default: enabled.",
    )
    p.add_argument(
        "--tight-axis-min-span",
        type=float,
        default=0.05,
        help="Minimum y-axis span for tight 0-1 score plots. Default: 0.05.",
    )
    p.add_argument(
        "--tight-axis-pad-fraction",
        type=float,
        default=0.15,
        help="Padding fraction for tight y-axis ranges. Default: 0.15.",
    )
    p.add_argument(
        "--tight-axis-min-pad",
        type=float,
        default=0.005,
        help="Minimum padding for tight y-axis ranges. Default: 0.005.",
    )
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
    p.add_argument(
        "--include-esa-leakage",
        action="store_true",
        help="Export detailed ESA leakage-by-class poster figure.",
    )
    p.add_argument(
        "--esa-leakage-metric",
        choices=["pred_mass_sum", "pred_mass_share"],
        default="pred_mass_sum",
        help="ESA class-wise leakage metric to plot. Default: pred_mass_sum.",
    )
    p.add_argument(
        "--esa-leakage-mode",
        choices=["grouped_stacked"],
        default="grouped_stacked",
        help="ESA leakage plot mode. Currently only grouped_stacked is implemented.",
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


def tight_unit_ylim(
    values: Sequence[float],
    min_span: float = 0.05,
    pad_fraction: float = 0.15,
    min_pad: float = 0.005,
) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    ymin_data = float(np.min(vals))
    ymax_data = float(np.max(vals))
    data_span = ymax_data - ymin_data
    pad = max(min_pad, pad_fraction * data_span)
    ymin = max(0.0, ymin_data - pad)
    ymax = min(1.0, ymax_data + pad)

    if ymax - ymin < min_span:
        center = 0.5 * (ymin + ymax)
        ymin = center - 0.5 * min_span
        ymax = center + 0.5 * min_span
        if ymin < 0.0:
            ymax = min(1.0, ymax - ymin)
            ymin = 0.0
        if ymax > 1.0:
            ymin = max(0.0, ymin - (ymax - 1.0))
            ymax = 1.0

    return ymin, ymax


def positive_corr_ylim(values: Sequence[float], tight: bool) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if not tight:
        return 0.0, 1.0
    return max(0.0, float(np.min(vals)) - 0.05), min(1.0, float(np.max(vals)) + 0.05)


def add_bar_labels(ax, bars, fmt: str = "{:.3f}") -> None:
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


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
            esa_class = row.get("esa_class", np.nan)
            if pd.notna(esa_class):
                esa_class = int(esa_class)
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
                    "esa_class": esa_class,
                    "esa_class_label": row.get("esa_class_label", ""),
                    "group": row.get("group", ""),
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
    bar_tight_axis: bool,
    bar_labels: bool,
    selected: List[dict],
) -> None:
    stem = "poster_validation_viirs_correlation"
    metrics = [("spearman_log1p", "Spearman")]
    if include_secondary:
        metrics.append(("pearson_log1p", "Pearson"))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(len(models), dtype=float)
    width = 0.68 / len(metrics)
    all_values: List[float] = []
    for i, (metric, label) in enumerate(metrics):
        vals = metric_values(viirs, "correlation", metric, models)
        all_values.extend(vals)
        offset = (i - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=label)
        if bar_labels:
            add_bar_labels(ax, bars)
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
        ylim=positive_corr_ylim(all_values, bar_tight_axis),
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
    bar_tight_axis: bool,
    bar_labels: bool,
    tight_axis_min_span: float,
    tight_axis_pad_fraction: float,
    tight_axis_min_pad: float,
    selected: List[dict],
) -> None:
    stem = "poster_validation_gaia_binary_agreement"
    metrics = [("iou", "IoU")]
    if include_secondary:
        metrics.append(("f1", "F1"))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(len(models), dtype=float)
    width = 0.68 / len(metrics)
    all_values: List[float] = []
    for i, (metric, label) in enumerate(metrics):
        vals = metric_values(gaia, "prevalence_matched_binary", metric, models)
        all_values.extend(vals)
        offset = (i - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=label)
        if bar_labels:
            add_bar_labels(ax, bars)
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
        "Score" if include_secondary else "IoU",
        "GAIA 2019 impervious: prevalence-matched agreement",
        ylim=(
            tight_unit_ylim(all_values, tight_axis_min_span, tight_axis_pad_fraction, tight_axis_min_pad)
            if bar_tight_axis
            else (0, 1)
        ),
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
    bar_tight_axis: bool,
    bar_labels: bool,
    tight_axis_min_span: float,
    tight_axis_pad_fraction: float,
    tight_axis_min_pad: float,
    selected: List[dict],
) -> None:
    stem = "poster_validation_esa_topk_overlap"
    vals = metric_values(esa, "topk", "topk_esa_built_share", models, topk_frac=esa_topk_frac)
    rows = selected_rows(esa, "topk", "topk_esa_built_share", models, topk_frac=esa_topk_frac)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    bars = ax.bar(np.arange(len(models)), vals, width=0.65)
    if bar_labels:
        add_bar_labels(ax, bars)
    setup_bar_axis(
        ax,
        labels,
        "Share of top predicted cells in ESA built-up",
        f"ESA WorldCover 2020: top-{percent_label(esa_topk_frac)} overlap with built-up class",
        ylim=(
            tight_unit_ylim(vals, tight_axis_min_span, tight_axis_pad_fraction, tight_axis_min_pad)
            if bar_tight_axis
            else (0, 1)
        ),
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
    bar_labels: bool,
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
    bars = ax.bar(np.arange(len(models)), vals, width=0.65)
    if bar_labels:
        add_bar_labels(ax, bars)
    setup_bar_axis(ax, labels, ylabel, title, ylim=(0, max(0.01, max(vals) * 1.15)))
    save_figure(fig, out_dir, stem, formats, dpi)
    append_selected(selected, source, stem, rows, models, labels, notes)


def esa_leakage_rows(esa: pd.DataFrame, metric: str, models: Sequence[str]) -> pd.DataFrame:
    required = {"model", "metric_group", "metric", "value", "esa_class"}
    missing_cols = sorted(required - set(esa.columns))
    if missing_cols:
        raise ValueError(f"ESA CSV is missing required leakage columns: {missing_cols}")

    expected_classes = ESA_LEAKAGE_HARD_CLASSES + ESA_LEAKAGE_SOFT_CLASSES
    sub = esa[
        (esa["metric_group"] == "hard_nonbuilt_by_class")
        & (esa["metric"] == metric)
        & (pd.to_numeric(esa["esa_class"], errors="coerce").isin(expected_classes))
    ].copy()

    if sub.empty:
        raise ValueError(
            "ESA leakage rows not found. Expected metric_group='hard_nonbuilt_by_class' "
            "and metric='pred_mass_sum' or 'pred_mass_share'. Re-run "
            "09_evaluate_against_esa_worldcover.py to regenerate the ESA metrics CSV."
        )

    missing_models = [model for model in models if model not in set(sub["model"].astype(str))]
    if missing_models:
        raise ValueError(f"ESA leakage rows are missing model(s): {missing_models}")

    sub["model"] = sub["model"].astype(str)
    sub["esa_class"] = pd.to_numeric(sub["esa_class"], errors="coerce").astype("Int64")
    sub["value"] = numeric_series(sub["value"], "value")
    sub["esa_class_label"] = sub.apply(
        lambda row: (
            row["esa_class_label"]
            if "esa_class_label" in sub.columns and pd.notna(row.get("esa_class_label")) and str(row.get("esa_class_label"))
            else ESA_LEAKAGE_CLASS_LABELS.get(int(row["esa_class"]), str(row["esa_class"]))
        ),
        axis=1,
    )
    sub["group"] = sub["esa_class"].apply(lambda code: "hard" if int(code) in ESA_LEAKAGE_HARD_CLASSES else "soft")
    return sub


def make_esa_leakage_by_class(
    esa: pd.DataFrame,
    out_dir: Path,
    models: Sequence[str],
    labels: Sequence[str],
    formats: Sequence[str],
    dpi: int,
    metric: str,
    selected: List[dict],
) -> None:
    stem = "poster_validation_esa_leakage_by_class"
    sub = esa_leakage_rows(esa, metric, models)
    expected_classes = ESA_LEAKAGE_HARD_CLASSES + ESA_LEAKAGE_SOFT_CLASSES
    available_classes = set(sub["esa_class"].dropna().astype(int).tolist())
    missing_classes = [code for code in expected_classes if code not in available_classes]
    if missing_classes:
        print(f"[WARN] ESA leakage classes absent from CSV and plotted as zero: {missing_classes}")

    values = {
        model: {code: 0.0 for code in expected_classes}
        for model in models
    }
    row_lookup = {}
    for _, row in sub.iterrows():
        model = str(row["model"])
        code = int(row["esa_class"])
        value = float(row["value"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite ESA leakage value for model={model}, esa_class={code}, metric={metric}.")
        values[model][code] = value
        row_lookup[(model, code)] = row

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(models), dtype=float)
    width = 0.32
    hard_x = x - width / 2.0
    soft_x = x + width / 2.0
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        code: color_cycle[i % len(color_cycle)] if color_cycle else None
        for i, code in enumerate(expected_classes)
    }

    max_stack = 0.0
    for bar_group, class_codes, xpos in [
        ("Hard", ESA_LEAKAGE_HARD_CLASSES, hard_x),
        ("Soft", ESA_LEAKAGE_SOFT_CLASSES, soft_x),
    ]:
        bottoms = np.zeros(len(models), dtype=float)
        for code in class_codes:
            heights = np.array([values[model][code] for model in models], dtype=float)
            ax.bar(
                xpos,
                heights,
                width=width,
                bottom=bottoms,
                label=ESA_LEAKAGE_CLASS_LABELS.get(code, str(code)),
                color=colors.get(code),
            )
            bottoms += heights
        max_stack = max(max_stack, float(np.max(bottoms)) if bottoms.size else 0.0)
        for xpos_i, total in zip(xpos, bottoms):
            ax.annotate(
                bar_group,
                xy=(xpos_i, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if metric == "pred_mass_sum":
        ax.set_title("ESA WorldCover 2020: predicted mass outside built-up class")
        ax.set_ylabel("Predicted built-up mass outside ESA built-up")
    else:
        ax.set_title("ESA WorldCover 2020: share of predicted mass outside built-up class")
        ax.set_ylabel("Share of predicted mass outside ESA built-up")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(0.01, max_stack * 1.18))
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_figure(fig, out_dir, stem, formats, dpi)

    selected_rows_for_csv = []
    for model in models:
        for code in expected_classes:
            if (model, code) in row_lookup:
                row = row_lookup[(model, code)].copy()
            else:
                row = pd.Series(
                    {
                        "model": model,
                        "metric_group": "hard_nonbuilt_by_class",
                        "metric": metric,
                        "value": 0.0,
                        "esa_class": code,
                        "esa_class_label": ESA_LEAKAGE_CLASS_LABELS.get(code, str(code)),
                        "group": "hard" if code in ESA_LEAKAGE_HARD_CLASSES else "soft",
                    }
                )
            selected_rows_for_csv.append(row)
    append_selected(
        selected,
        "ESA",
        stem,
        pd.DataFrame(selected_rows_for_csv),
        models,
        labels,
        "Predicted built-up mass assigned to non-built ESA WorldCover classes.",
    )


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
        "esa_class",
        "esa_class_label",
        "group",
        "notes",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_df = pd.DataFrame(selected, columns=columns)
    selected_df["esa_class"] = selected_df["esa_class"].apply(
        lambda value: "" if pd.isna(value) else str(int(float(value)))
    )
    selected_df.to_csv(out_dir / "poster_validation_selected_metrics.csv", index=False)


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
        args.bar_tight_axis,
        args.bar_labels,
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
        args.bar_tight_axis,
        args.bar_labels,
        args.tight_axis_min_span,
        args.tight_axis_pad_fraction,
        args.tight_axis_min_pad,
        selected,
    )
    make_esa_topk_overlap(
        esa,
        out_dir,
        models,
        labels,
        formats,
        args.dpi,
        args.esa_topk_frac,
        args.bar_tight_axis,
        args.bar_labels,
        args.tight_axis_min_span,
        args.tight_axis_pad_fraction,
        args.tight_axis_min_pad,
        selected,
    )
    if args.include_leakage:
        make_leakage_figure(esa, gaia, out_dir, models, labels, formats, args.dpi, args.bar_labels, selected)
    if args.include_esa_leakage:
        make_esa_leakage_by_class(
            esa,
            out_dir,
            models,
            labels,
            formats,
            args.dpi,
            args.esa_leakage_metric,
            selected,
        )

    write_selected_metrics(selected, out_dir)
    print(f"[INFO] Saved poster validation figures and selected metrics to: {out_dir}")


if __name__ == "__main__":
    main()
