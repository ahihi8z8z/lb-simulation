#!/usr/bin/env python3
"""Unified plotting tool for sweep CSV line and grouped bar charts."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

POLICY_COL = "policy"
SCENARIO_COL = "scenario"
STATUS_COL = "status"
X_SCALE_CANDIDATES = [
    "service_class.classes[1].scale_beta",
    "service_class.classes[0].scale_beta",
]
X_NORM_COL = "__x_normalized__"
X_STD_COL = "__scenario_std__"
REMOVE_POLICIES = {"min_ema_latency", "latency_p2c", "sp-wrr"}
DEFAULT_FIG_FACE = "#d9d9d9"
DEFAULT_AX_FACE = "#e6e6e6"
DEFAULT_DPI = 150
DEFAULT_BAR_SCALE = 10.0
DEFAULT_BAR_STD = 0.41
DEFAULT_LINE_STD_SCALE = DEFAULT_BAR_SCALE
SERVICE_LABELS = {
    0: "Conversation service",
    1: "API service",
}


def _identity(value: float) -> float:
    return value


def _to_percent(value: float) -> float:
    return value * 100.0


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    ylabel: str
    transform: Callable[[float], float] = _identity
    scenario_name: str | None = None
    scenario_std: float | None = None
    std_scale: float | None = None


@dataclass(frozen=True)
class CompositeLineSpec:
    name: str
    panels: Sequence[MetricSpec]
    ncols: int


@dataclass(frozen=True)
class BarPanelSpec:
    title: str
    metric_col: str
    group_kind: str
    ylabel: str
    transform: Callable[[float], float] = _identity
    snapshot_scale: float | None = None
    snapshot_std: float | None = None
    snapshot_scenario: str | None = None


@dataclass(frozen=True)
class CompositeBarSpec:
    name: str
    panels: Sequence[BarPanelSpec]
    ncols: int


@dataclass(frozen=True)
class CompositePanelSpec:
    kind: str
    line: MetricSpec | None = None
    bar: BarPanelSpec | None = None


@dataclass(frozen=True)
class MixedCompositeSpec:
    name: str
    panels: Sequence[CompositePanelSpec]
    ncols: int


POLICY_STYLES = {
    "static-wrr": {
        "label": "S-WRR",
        "line": {
            "marker": "*",
            "color": "blue",
            "linestyle": ":",
            "linewidth": 1.5,
            "markersize": 4,
        },
        "bar": {
            "facecolor": "lightblue",
            "edgecolor": "black",
            "hatch": "////",
            "linewidth": 1.2,
        },
    },
    "least_connection": {
        "label": "LC",
        "line": {
            "marker": "s",
            "color": "red",
            "linestyle": ":",
            "linewidth": 1.5,
            "markersize": 4,
        },
        "bar": {
            "facecolor": "lightcoral",
            "edgecolor": "black",
            "hatch": "\\\\\\\\",
            "linewidth": 1.2,
        },
    },
    "lp-wrr": {
        "label": "LP-WRR",
        "line": {
            "marker": "o",
            "color": "#f2b500",
            "linestyle": "-",
            "linewidth": 1.5,
            "markersize": 4,
        },
        "bar": {
            "facecolor": "#f2b500",
            "edgecolor": "black",
            "hatch": None,
            "linewidth": 1.2,
        },
    },
    "sp-wrr": {
        "label": "SP-WRR",
        "line": {
            "marker": "D",
            "color": "#2ca02c",
            "linestyle": "-.",
            "linewidth": 1.5,
            "markersize": 4,
        },
        "bar": {
            "facecolor": "#8fd19e",
            "edgecolor": "black",
            "hatch": "..",
            "linewidth": 1.2,
        },
    },
    "power_of_two_choices": {
        "label": "P2C",
        "line": {
            "marker": "^",
            "color": "gray",
            "linestyle": ":",
            "linewidth": 1.5,
            "markersize": 4,
        },
        "bar": {
            "facecolor": "lightgray",
            "edgecolor": "black",
            "hatch": "xx",
            "linewidth": 1.2,
        },
    },
}
FALLBACK_LINE_MARKERS = ["o", "s", "^", "D", "*", "X", "P", "v", "<", ">"]
FALLBACK_LINE_STYLES = ["-", "--", "-.", ":"]
FALLBACK_BAR_STYLES = [
    {
        "facecolor": "white",
        "edgecolor": "#2f5aa8",
        "hatch": "//////",
        "linewidth": 1.2,
    },
    {
        "facecolor": "#f2b500",
        "edgecolor": "black",
        "hatch": None,
        "linewidth": 1.2,
    },
    {
        "facecolor": "white",
        "edgecolor": "#8b0000",
        "hatch": "\\\\\\\\",
        "linewidth": 1.2,
    },
    {
        "facecolor": "white",
        "edgecolor": "#444444",
        "hatch": "xx",
        "linewidth": 1.2,
    },
]


LINE_SCALE_COMPOSITES = [
    CompositeLineSpec(
        name="scale",
        ncols=3,
        panels=(
            MetricSpec("mean_latency", "(a) Mean latency, std=0", "Latency (s)", scenario_std=0),
            MetricSpec("p95_latency", "(b) P95 latency, std=0", "Latency (s)", scenario_std=0),
            MetricSpec("outcomes.traffic.drop_rate", "(c) Drop ratio, std=0", "Drop ratio. (%)", transform=_to_percent, scenario_std=0),
            MetricSpec("mean_latency", "(d) Mean latency, std=0.41", "Latency (s)", scenario_std=0.41),
            MetricSpec("p95_latency", "(e) P95 latency, std=0.41", "Latency (s)", scenario_std=0.41),
            MetricSpec("outcomes.traffic.drop_rate", "(f) Drop ratio, std=0.41", "Drop ratio (%)", transform=_to_percent, scenario_std=0.41),
        ),
    ),
]


LINE_STD_COMPOSITES = [
    CompositeLineSpec(
        name="std",
        ncols=2,
        panels=(
            MetricSpec("mean_latency", "(a) Mean latency", "Latency (s)",std_scale=10),
            MetricSpec("p95_latency", "(b) P95 latency", "Latency (s)",std_scale=10),
        ),
    ),
]

BAR_COMPOSITES = [
    CompositeBarSpec(
        name="detail",
        ncols=3,
        panels=(
            BarPanelSpec(
                title="(a) Mean latency, std=0",
                metric_col="mean_latency",
                group_kind="service_latency_mean",
                ylabel="Latency (s)",
                snapshot_scale=10,
                snapshot_std=0,
            ),
            BarPanelSpec(
                title="(b) P95 latency, std=0",
                metric_col="p95_latency",
                group_kind="service_latency_p95",
                ylabel="Latency (s)",
                snapshot_scale=10,
                snapshot_std=0,
            ),
            BarPanelSpec(
                title="(c) Instance utilization, std=0",
                metric_col="outcomes.dispersion.worker_utilization_max_gap",
                group_kind="worker_utilization",
                ylabel="Instance utilization (%)",
                transform=_to_percent,
                snapshot_scale=10,
                snapshot_std=0,
            ),
            BarPanelSpec(
                title="(d) Mean latency, std=0.41",
                metric_col="mean_latency",
                group_kind="service_latency_mean",
                ylabel="Latency (s)",
                snapshot_scale=10,
                snapshot_std=0.41,
            ),
            BarPanelSpec(
                title="(e) P95 latency, std=0.41",
                metric_col="p95_latency",
                group_kind="service_latency_p95",
                ylabel="Latency (s)",
                snapshot_scale=10,
                snapshot_std=0.41,
            ),
            BarPanelSpec(
                title="(f) Instance utilization, std=0.41",
                metric_col="outcomes.dispersion.worker_utilization_max_gap",
                group_kind="worker_utilization",
                ylabel="Instance utilization (%)",
                transform=_to_percent,
                snapshot_scale=10,
                snapshot_std=0.41,
            ),
        ),
    ),
]
MIXED_COMPOSITES = []


def _sanitize_filename(raw: str) -> str:
    value = str(raw).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value or "unknown"


def _parse_std_from_scenario(raw: str) -> float:
    match = re.search(r"std-([0-9]*\.?[0-9]+)", str(raw).strip().lower())
    if not match:
        return float("nan")
    return float(match.group(1))


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if POLICY_COL not in df.columns or SCENARIO_COL not in df.columns:
        raise ValueError(f"Missing required columns: {POLICY_COL}, {SCENARIO_COL}")

    if STATUS_COL in df.columns:
        status = df[STATUS_COL].astype(str).str.strip().str.lower()
        df = df[status == "ok"].copy()
    else:
        df = df.copy()

    df[POLICY_COL] = df[POLICY_COL].astype(str).str.strip().replace("", "unknown")
    df[SCENARIO_COL] = df[SCENARIO_COL].astype(str).str.strip().replace("", "unknown")
    return df


def _pick_scale_column(columns: Iterable[str]) -> str | None:
    for candidate in X_SCALE_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _get_policies(df: pd.DataFrame) -> list[str]:
    policies = sorted(df[POLICY_COL].dropna().unique())
    return [policy for policy in policies if policy not in REMOVE_POLICIES]


def _policy_label(policy: str) -> str:
    return POLICY_STYLES.get(policy, {}).get("label", policy)


def _fallback_colors() -> list[str]:
    colors = plt.rcParams.get("axes.prop_cycle", None)
    color_values = colors.by_key().get("color", []) if colors is not None else []
    if color_values:
        return color_values
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def _resolve_line_style(policy: str, index: int) -> dict:
    colors = _fallback_colors()
    style = {
        "marker": FALLBACK_LINE_MARKERS[index % len(FALLBACK_LINE_MARKERS)],
        "color": colors[index % len(colors)],
        "linestyle": FALLBACK_LINE_STYLES[index % len(FALLBACK_LINE_STYLES)],
        "linewidth": 1.8,
        "markersize": 8,
    }
    style.update(POLICY_STYLES.get(policy, {}).get("line", {}))
    style.setdefault("markerfacecolor", style["color"])
    style.setdefault("markeredgecolor", style["color"])
    return style


def _resolve_bar_style(policy: str, index: int) -> dict:
    style = dict(FALLBACK_BAR_STYLES[index % len(FALLBACK_BAR_STYLES)])
    style.update(POLICY_STYLES.get(policy, {}).get("bar", {}))
    return style


def _apply_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor(DEFAULT_AX_FACE)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=8, length=3, width=0.8, colors="black")


def _iter_metric_columns(composites: Sequence[CompositeLineSpec]) -> list[str]:
    seen: list[str] = []
    for composite in composites:
        for panel in composite.panels:
            if panel.column not in seen:
                seen.append(panel.column)
    return seen


def _iter_mixed_line_metrics(
    composites: Sequence[MixedCompositeSpec], kind: str
) -> list[str]:
    seen: list[str] = []
    for composite in composites:
        for panel in composite.panels:
            if panel.kind != kind or panel.line is None:
                continue
            if panel.line.column not in seen:
                seen.append(panel.line.column)
    return seen


def _prepare_scale_frame(df: pd.DataFrame, metrics: Sequence[str]) -> tuple[pd.DataFrame, str] | None:
    scale_col = _pick_scale_column(df.columns)
    if scale_col is None:
        return None

    required = [POLICY_COL, SCENARIO_COL, scale_col, *metrics]
    available = [column for column in required if column in df.columns]
    if len(available) < 4:
        return None

    frame = df[[column for column in required if column in df.columns]].copy()
    frame[scale_col] = pd.to_numeric(frame[scale_col], errors="coerce")
    frame = frame[frame[scale_col] > 0].copy()
    for metric in metrics:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    usable_metrics = [metric for metric in metrics if metric in frame.columns]
    if not usable_metrics:
        return None

    frame = frame.dropna(subset=[scale_col, *usable_metrics]).copy()
    if frame.empty:
        return None

    frame["_inverse_scale"] = 1.0 / frame[scale_col]
    x_min = float(frame["_inverse_scale"].min())
    if x_min <= 0:
        return None
    frame[X_NORM_COL] = frame["_inverse_scale"] / x_min
    return frame, scale_col


def _prepare_std_frame(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame | None:
    scale_col = _pick_scale_column(df.columns)
    required = [POLICY_COL, SCENARIO_COL, *metrics]
    if scale_col is not None:
        required.append(scale_col)
    available = [column for column in required if column in df.columns]
    if len(available) < 3:
        return None

    frame = df[[column for column in required if column in df.columns]].copy()
    frame[X_STD_COL] = frame[SCENARIO_COL].map(_parse_std_from_scenario)
    if scale_col is not None and scale_col in frame.columns:
        frame[scale_col] = pd.to_numeric(frame[scale_col], errors="coerce")
    for metric in metrics:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    usable_metrics = [metric for metric in metrics if metric in frame.columns]
    if not usable_metrics:
        return None

    frame = frame.dropna(subset=[X_STD_COL, *usable_metrics]).copy()
    if frame.empty:
        return None
    return frame


def _register_legend_entry(
    legend_handles: dict[str, object], label: str, handle: object
) -> None:
    if label not in legend_handles:
        legend_handles[label] = handle


def _mark_no_data(
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str,
    xlabel: str | None = None,
    grid: bool = False,
) -> None:
    ax.text(
        0.5,
        0.5,
        "No data",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.grid(grid)


def _finalize_figure(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    output_path: Path,
    legend_handles: dict[str, object],
    ncols: int,
    shared_row_xlabel: str | None = None,
    row_ncols: int | None = None,
    panel_count: int | None = None,
    shared_xlabel_bottom_pad: float | None = None,
    shared_xlabel_offset: float | None = None,
    last_row_shared_xlabel_offset: float | None = None,
    row_spacing: float | None = None,
) -> None:
    fig.patch.set_facecolor(DEFAULT_FIG_FACE)
    for ax in axes:
        _apply_axis_style(ax)

    legend_cols = 0
    legend_rows = 0
    if legend_handles:
        width_based_cols = max(1, int(fig.get_size_inches()[0] / 1.9))
        legend_cols = max(1, min(ncols, len(legend_handles), width_based_cols))
        legend_rows = math.ceil(len(legend_handles) / legend_cols)

    top_pad = 0.02 + (0.055 * legend_rows if legend_rows else 0.0)
    bottom_pad = shared_xlabel_bottom_pad if shared_xlabel_bottom_pad is not None else (
        0.08 if shared_row_xlabel else 0.02
    )
    fig.tight_layout(rect=(0.02, bottom_pad, 0.98, 1.0 - top_pad))

    if row_spacing is not None:
        fig.subplots_adjust(hspace=row_spacing)

    if shared_row_xlabel and row_ncols and panel_count:
        visible_axes = [ax for ax in axes[:panel_count] if ax.get_visible()]
        row_groups = [
            visible_axes[start:start + row_ncols]
            for start in range(0, len(visible_axes), row_ncols)
        ]
        inner_row_label_offset = (
            shared_xlabel_offset
            if shared_xlabel_offset is not None
            else (0.04 if len(row_groups) > 1 else 0.05)
        )
        last_row_label_offset = (
            last_row_shared_xlabel_offset
            if last_row_shared_xlabel_offset is not None
            else 0.04
        )
        for row_index, row_axes in enumerate(row_groups):
            if not row_axes:
                continue
            left = min(ax.get_position().x0 for ax in row_axes)
            right = max(ax.get_position().x1 for ax in row_axes)
            bottom = min(ax.get_position().y0 for ax in row_axes)
            label_offset = (
                last_row_label_offset
                if row_index == len(row_groups) - 1
                else inner_row_label_offset
            )
            y_pos = max(0.015, bottom - label_offset)
            fig.text(
                (left + right) / 2.0,
                y_pos,
                shared_row_xlabel,
                ha="center",
                va="top",
                fontsize=10,
            )

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0 - top_pad + 0.004),
            ncol=legend_cols,
            fontsize=7.5,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor="black",
            borderpad=0.28,
            handlelength=2.5,
            handletextpad=0.6,
            columnspacing=1.2,
            borderaxespad=0.0,
        )
    fig.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(fig)


def _merge_policies(*frames: pd.DataFrame | None) -> list[str]:
    policies: list[str] = []
    for frame in frames:
        if frame is None:
            continue
        for policy in _get_policies(frame):
            if policy not in policies:
                policies.append(policy)
    return policies


def _resolve_scenario_name(
    frame: pd.DataFrame | None,
    *,
    scenario_name: str | None = None,
    scenario_std: float | None = None,
    fallback_std: float | None = None,
) -> str | None:
    if frame is None or frame.empty:
        return None

    scenarios = sorted(frame[SCENARIO_COL].dropna().unique())
    if not scenarios:
        return None

    if scenario_name is not None and scenario_name in scenarios:
        return scenario_name

    std_candidates: list[float] = []
    if scenario_std is not None:
        std_candidates.append(float(scenario_std))
    if fallback_std is not None:
        std_candidates.append(float(fallback_std))

    for target_std in std_candidates:
        preferred = f"std-{target_std:.2f}"
        if preferred in scenarios:
            return preferred

        for scenario in scenarios:
            parsed = _parse_std_from_scenario(scenario)
            if not math.isnan(parsed) and abs(parsed - target_std) < 1e-9:
                return scenario

    if scenario_name is not None or scenario_std is not None:
        return None
    return scenarios[0]


def _select_scale_panel_frame(
    frame: pd.DataFrame | None,
    panel: MetricSpec,
    *,
    fallback_std: float | None = None,
) -> tuple[str | None, pd.DataFrame | None]:
    scenario_name = _resolve_scenario_name(
        frame,
        scenario_name=panel.scenario_name,
        scenario_std=panel.scenario_std,
        fallback_std=fallback_std,
    )
    if frame is None or scenario_name is None:
        return scenario_name, None

    panel_frame = frame[frame[SCENARIO_COL] == scenario_name].copy()
    if panel_frame.empty:
        return scenario_name, None
    return scenario_name, panel_frame


def _sparsify_ticks(values: Sequence[float], max_ticks: int = 5) -> list[float]:
    ticks = list(values)
    if len(ticks) <= max_ticks:
        return ticks

    positions = {
        round(index * (len(ticks) - 1) / (max_ticks - 1))
        for index in range(max_ticks)
    }
    return [ticks[index] for index in sorted(positions)]


def _integer_scale_ticks(values: Sequence[float], max_ticks: int = 10) -> list[float]:
    sorted_values = sorted(set(values))
    if not sorted_values:
        return []

    min_int = math.ceil(min(sorted_values))
    max_int = math.floor(max(sorted_values))
    if min_int <= max_int:
        integer_ticks = [float(value) for value in range(min_int, max_int + 1)]
    else:
        integer_ticks = [float(round(sorted_values[0]))]
    if len(integer_ticks) <= max_ticks:
        return integer_ticks

    positions = {
        round(index * (len(integer_ticks) - 1) / (max_ticks - 1))
        for index in range(max_ticks)
    }
    return [integer_ticks[index] for index in sorted(positions)]


def _plot_scale_line_panel(
    ax: plt.Axes,
    scenario_frame: pd.DataFrame,
    policies: Sequence[str],
    panel: MetricSpec,
    legend_handles: dict[str, object],
    *,
    show_xlabel: bool = True,
) -> bool:
    if panel.column not in scenario_frame.columns:
        _mark_no_data(
            ax,
            title=panel.title,
            ylabel=panel.ylabel,
            xlabel="API Traffic scale" if show_xlabel else None,
            grid=False,
        )
        return False

    x_ticks = _integer_scale_ticks(scenario_frame[X_NORM_COL].dropna().unique(), max_ticks=10)
    plotted = False
    for policy_index, policy in enumerate(policies):
        part = scenario_frame[scenario_frame[POLICY_COL] == policy]
        if part.empty:
            continue

        series = (
            part[[X_NORM_COL, panel.column]]
            .dropna()
            .groupby(X_NORM_COL, as_index=False)[panel.column]
            .min()
            .sort_values(X_NORM_COL, ascending=False)
        )
        if series.empty:
            continue

        style = _resolve_line_style(policy, policy_index)
        x_values = series[X_NORM_COL].tolist()
        y_values = [panel.transform(value) for value in series[panel.column].tolist()]
        line, = ax.plot(
            x_values,
            y_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["markeredgecolor"],
            linewidth=float(style["linewidth"]),
            markersize=float(style["markersize"]),
            label=_policy_label(policy),
        )
        _register_legend_entry(legend_handles, _policy_label(policy), line)
        plotted = True

    ax.set_title(panel.title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("API Traffic scale")
    ax.set_ylabel(panel.ylabel)
    if x_ticks:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{int(round(tick))}x" for tick in x_ticks])
    ax.grid(True, alpha=0.25)

    if not plotted:
        _mark_no_data(
            ax,
            title=panel.title,
            ylabel=panel.ylabel,
            xlabel="API Traffic scale" if show_xlabel else None,
            grid=False,
        )
    return plotted


def _plot_std_line_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    policies: Sequence[str],
    panel: MetricSpec,
    legend_handles: dict[str, object],
    *,
    show_xlabel: bool = True,
) -> bool:
    if panel.column not in frame.columns:
        _mark_no_data(
            ax,
            title=panel.title,
            ylabel=panel.ylabel,
            xlabel="Instance Capacity Std" if show_xlabel else None,
            grid=False,
        )
        return False

    working_frame = frame
    scale_col = _pick_scale_column(working_frame.columns)
    if panel.std_scale is not None:
        if scale_col is None:
            _mark_no_data(
                ax,
                title=panel.title,
                ylabel=panel.ylabel,
                xlabel="Instance Capacity Std" if show_xlabel else None,
                grid=False,
            )
            return False
        working_frame = working_frame[
            working_frame[scale_col].sub(float(panel.std_scale)).abs() < 1e-9
        ].copy()
        if working_frame.empty:
            _mark_no_data(
                ax,
                title=panel.title,
                ylabel=panel.ylabel,
                xlabel="Instance Capacity Std" if show_xlabel else None,
                grid=False,
            )
            return False

    plotted = False
    for policy_index, policy in enumerate(policies):
        part = working_frame[working_frame[POLICY_COL] == policy]
        if part.empty:
            continue

        series = (
            part[[X_STD_COL, panel.column]]
            .dropna()
            .groupby(X_STD_COL, as_index=False)[panel.column]
            .min()
            .sort_values(X_STD_COL)
        )
        if series.empty:
            continue

        style = _resolve_line_style(policy, policy_index)
        x_values = series[X_STD_COL].tolist()
        y_values = [panel.transform(value) for value in series[panel.column].tolist()]
        line, = ax.plot(
            x_values,
            y_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["markeredgecolor"],
            linewidth=float(style["linewidth"]),
            markersize=float(style["markersize"]),
            label=_policy_label(policy),
        )
        _register_legend_entry(legend_handles, _policy_label(policy), line)
        plotted = True

    ax.set_title(panel.title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Instance Capacity Std")
    ax.set_ylabel(panel.ylabel)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.25)

    if not plotted:
        _mark_no_data(
            ax,
            title=panel.title,
            ylabel=panel.ylabel,
            xlabel="Instance Capacity Std" if show_xlabel else None,
            grid=False,
        )
    return plotted


def _render_line_composites_by_scale(
    df: pd.DataFrame, csv_path: Path, output_dir: Path
) -> list[Path]:
    metrics = _iter_metric_columns(LINE_SCALE_COMPOSITES)
    prepared = _prepare_scale_frame(df, metrics)
    if prepared is None:
        return []

    frame, _scale_col = prepared
    policies = _get_policies(frame)
    if not policies:
        return []
    scenarios = sorted(frame[SCENARIO_COL].dropna().unique())
    output_paths: list[Path] = []

    for composite in LINE_SCALE_COMPOSITES:
        explicit_selector_mode = any(
            panel.scenario_name is not None or panel.scenario_std is not None
            for panel in composite.panels
        )

        if explicit_selector_mode:
            panels = [panel for panel in composite.panels if panel.column in frame.columns]
            if not panels:
                continue

            panel_frames_by_index: dict[int, tuple[str, pd.DataFrame] | None] = {}
            selected_frames: list[pd.DataFrame] = []
            used_scenarios: list[str] = []

            for idx, panel in enumerate(panels):
                scenario_name, panel_frame = _select_scale_panel_frame(frame, panel)
                if scenario_name is not None and panel_frame is not None:
                    panel_frames_by_index[idx] = (scenario_name, panel_frame)
                    selected_frames.append(panel_frame)
                    used_scenarios.append(scenario_name)

            selected_policies = _merge_policies(*selected_frames)
            if not selected_policies:
                continue

            rows = math.ceil(len(panels) / composite.ncols)
            fig, axes_grid = plt.subplots(
                rows,
                composite.ncols,
                figsize=(5.0 * composite.ncols, 2.8 * rows),
                squeeze=False,
            )
            axes = axes_grid.flatten()
            legend_handles: dict[str, object] = {}

            for idx, panel in enumerate(panels):
                ax = axes[idx]
                selected = panel_frames_by_index.get(idx)
                if selected is None:
                    _mark_no_data(
                        ax,
                        title=panel.title,
                        ylabel=panel.ylabel,
                        xlabel=None,
                        grid=False,
                    )
                    continue
                _, panel_frame = selected
                _plot_scale_line_panel(
                    ax,
                    panel_frame,
                    selected_policies,
                    panel,
                    legend_handles,
                    show_xlabel=False,
                )

            for unused_ax in axes[len(panels):]:
                unused_ax.set_visible(False)

            scenario_suffix = (
                _sanitize_filename(used_scenarios[0])
                if len(set(used_scenarios)) == 1 and used_scenarios
                else "mixed_scenarios"
            )
            output_path = output_dir / (
                f"{csv_path.stem}_{composite.name}_{scenario_suffix}.pdf"
            )
            _finalize_figure(
                fig,
                [ax for ax in axes[: len(panels)] if ax.get_visible()],
                output_path,
                legend_handles,
                ncols=len(selected_policies),
                shared_row_xlabel="API Traffic scale",
                row_ncols=composite.ncols,
                panel_count=len(panels),
                shared_xlabel_bottom_pad=0.03,
                shared_xlabel_offset=0.055,
                last_row_shared_xlabel_offset=0.055,
                row_spacing=0.5,
            )
            output_paths.append(output_path)
            continue

        for scenario in scenarios:
            scenario_frame = frame[frame[SCENARIO_COL] == scenario].copy()
            if scenario_frame.empty:
                continue

            panels = [panel for panel in composite.panels if panel.column in scenario_frame.columns]
            if not panels:
                continue

            rows = math.ceil(len(panels) / composite.ncols)
            fig, axes_grid = plt.subplots(
                rows,
                composite.ncols,
                figsize=(5.0 * composite.ncols, 2.8 * rows),
                squeeze=False,
            )
            axes = axes_grid.flatten()
            legend_handles: dict[str, object] = {}

            for idx, panel in enumerate(panels):
                ax = axes[idx]
                _plot_scale_line_panel(
                    ax,
                    scenario_frame,
                    policies,
                    panel,
                    legend_handles,
                    show_xlabel=False,
                )

            for unused_ax in axes[len(panels):]:
                unused_ax.set_visible(False)

            output_path = output_dir / (
                f"{csv_path.stem}_{composite.name}_{_sanitize_filename(scenario)}.pdf"
            )
            _finalize_figure(
                fig,
                [ax for ax in axes[: len(panels)] if ax.get_visible()],
                output_path,
                legend_handles,
                ncols=len(selected_policies),
                shared_row_xlabel="API Traffic scale",
                row_ncols=composite.ncols,
                panel_count=len(panels),
                shared_xlabel_bottom_pad=0.10,
                shared_xlabel_offset=0.055,
                row_spacing=0.5,
            )
            output_paths.append(output_path)
    return output_paths


def _render_line_composites_by_std(
    df: pd.DataFrame, csv_path: Path, output_dir: Path
) -> list[Path]:
    metrics = _iter_metric_columns(LINE_STD_COMPOSITES)
    frame = _prepare_std_frame(df, metrics)
    if frame is None:
        return []

    policies = _get_policies(frame)
    if not policies:
        return []
    output_paths: list[Path] = []

    for composite in LINE_STD_COMPOSITES:
        panels = [panel for panel in composite.panels if panel.column in frame.columns]
        if not panels:
            continue

        rows = math.ceil(len(panels) / composite.ncols)
        fig, axes_grid = plt.subplots(
            rows,
            composite.ncols,
            figsize=(4.0 * composite.ncols, 2.8 * rows),
            squeeze=False,
        )
        axes = axes_grid.flatten()
        legend_handles: dict[str, object] = {}

        for idx, panel in enumerate(panels):
            ax = axes[idx]
            _plot_std_line_panel(
                ax,
                frame,
                policies,
                panel,
                legend_handles,
                show_xlabel=False,
            )

        for unused_ax in axes[len(panels):]:
            unused_ax.set_visible(False)

        output_path = output_dir / f"{csv_path.stem}_{composite.name}.pdf"
        _finalize_figure(
            fig,
            [ax for ax in axes[: len(panels)] if ax.get_visible()],
            output_path,
            legend_handles,
            ncols=len(policies),
            shared_row_xlabel="Instance Capacity Variability",
            row_ncols=composite.ncols,
            panel_count=len(panels),
            shared_xlabel_bottom_pad=0.05,
            last_row_shared_xlabel_offset =0.09,
            row_spacing=0.5,
        )
        output_paths.append(output_path)

    return output_paths


def _discover_indexed_columns(
    columns: Iterable[str], prefix: str, suffix: str
) -> list[tuple[int, str]]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    indexed: list[tuple[int, str]] = []
    for column in columns:
        match = pattern.match(column)
        if match:
            indexed.append((int(match.group(1)), column))
    indexed.sort(key=lambda item: item[0])
    return indexed


def _bar_group_definition(
    df: pd.DataFrame, group_kind: str
) -> tuple[list[str], list[str]] | None:
    if group_kind == "worker_utilization":
        indexed = _discover_indexed_columns(df.columns, "breakdown.worker.utilization.worker_", "")
        columns = [column for _, column in indexed]
        labels = [f"Instance-{index + 1}" for index, _ in indexed]
        return (columns, labels) if columns else None

    worker_metric_suffix = {
        "worker_latency_mean": ".mean",
        "worker_latency_p95": ".p95",
        "worker_latency_p99": ".p99",
        "worker_drop": ".drop_rate",
    }
    if group_kind in worker_metric_suffix:
        prefix = (
            "breakdown.worker.drop.worker_"
            if group_kind == "worker_drop"
            else "breakdown.worker.latency.worker_"
        )
        indexed = _discover_indexed_columns(df.columns, prefix, worker_metric_suffix[group_kind])
        columns = [column for _, column in indexed]
        labels = [f"Instance-{index + 1}" for index, _ in indexed]
        return (columns, labels) if columns else None

    service_metric_suffix = {
        "service_latency_mean": ".mean",
        "service_latency_p95": ".p95",
        "service_latency_p99": ".p99",
        "service_drop": ".drop_rate",
    }
    if group_kind in service_metric_suffix:
        prefix = (
            "breakdown.service.drop.class_"
            if group_kind == "service_drop"
            else "breakdown.service.latency.class_"
        )
        indexed = _discover_indexed_columns(df.columns, prefix, service_metric_suffix[group_kind])
        columns = [column for _, column in indexed]
        labels = [SERVICE_LABELS.get(index, f"Class-{index}") for index, _ in indexed]
        overall_col = {
            "service_latency_mean": "mean_latency",
            "service_latency_p95": "p95_latency",
            "service_latency_p99": "p99_latency",
            "service_drop": "outcomes.traffic.drop_rate",
        }[group_kind]
        if overall_col in df.columns:
            columns.append(overall_col)
            labels.append("System overall")
        return (columns, labels) if columns else None

    return None


def _prepare_bar_frame(
    df: pd.DataFrame,
    *,
    scale: float | None = DEFAULT_BAR_SCALE,
    std: float | None = DEFAULT_BAR_STD,
    scenario_name: str | None = None,
) -> pd.DataFrame | None:
    scale_col = _pick_scale_column(df.columns)
    if scale_col is None:
        return None

    frame = df.copy()
    frame[scale_col] = pd.to_numeric(frame[scale_col], errors="coerce")
    frame[X_STD_COL] = frame[SCENARIO_COL].map(_parse_std_from_scenario)
    if scale is not None:
        frame = frame[frame[scale_col].sub(scale).abs() < 1e-9].copy()
    if scenario_name is not None:
        frame = frame[frame[SCENARIO_COL] == scenario_name].copy()
    elif std is not None:
        frame = frame[frame[X_STD_COL].sub(std).abs() < 1e-9].copy()
    if frame.empty:
        return None
    return frame


def _best_row_per_policy(df: pd.DataFrame, metric_col: str, policies: Sequence[str]) -> pd.DataFrame:
    if metric_col not in df.columns:
        return df.iloc[0:0].copy()

    frame = df[df[POLICY_COL].isin(policies)].copy()
    frame[metric_col] = pd.to_numeric(frame[metric_col], errors="coerce")
    frame = frame.dropna(subset=[metric_col]).copy()
    if frame.empty:
        return frame

    idx = frame.groupby(POLICY_COL)[metric_col].idxmin()
    best = frame.loc[idx].copy()
    best["_policy_order"] = best[POLICY_COL].map({policy: index for index, policy in enumerate(policies)})
    best = best.sort_values("_policy_order")
    return best


def _prepare_bar_frame_for_panel(df: pd.DataFrame, panel: BarPanelSpec) -> pd.DataFrame | None:
    scale = panel.snapshot_scale if panel.snapshot_scale is not None else DEFAULT_BAR_SCALE
    std = panel.snapshot_std if panel.snapshot_std is not None else DEFAULT_BAR_STD
    if panel.snapshot_scenario is not None:
        std = None
    return _prepare_bar_frame(
        df,
        scale=scale,
        std=std,
        scenario_name=panel.snapshot_scenario,
    )


def _prepare_bar_panel_data(
    frame: pd.DataFrame,
    policies: Sequence[str],
    panel: BarPanelSpec,
) -> tuple[pd.DataFrame, list[str], list[str]] | None:
    definition = _bar_group_definition(frame, panel.group_kind)
    if definition is None:
        return None

    group_cols, group_labels = definition
    paired_columns = [
        (column, label)
        for column, label in zip(group_cols, group_labels)
        if column in frame.columns
    ]
    if not paired_columns:
        return None

    best = _best_row_per_policy(frame, panel.metric_col, policies)
    if best.empty:
        return None

    resolved_group_cols = [column for column, _ in paired_columns]
    resolved_group_labels = [label for _, label in paired_columns]
    numeric_cols = [panel.metric_col, *resolved_group_cols]
    for column in numeric_cols:
        best[column] = pd.to_numeric(best[column], errors="coerce")
    best = best.dropna(subset=[panel.metric_col, *resolved_group_cols]).copy()
    if best.empty:
        return None

    return best, resolved_group_cols, resolved_group_labels


def _plot_bar_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    policies: Sequence[str],
    panel: BarPanelSpec,
    legend_handles: dict[str, object],
) -> bool:
    prepared = _prepare_bar_panel_data(frame, policies, panel)
    if prepared is None:
        _mark_no_data(ax, title=panel.title, ylabel=panel.ylabel, grid=False)
        return False

    best, group_cols, group_labels = prepared
    x_values = list(range(len(group_cols)))
    width = 0.75 / max(len(best), 1)

    for policy_index, (_, row) in enumerate(best.iterrows()):
        policy = row[POLICY_COL]
        label = _policy_label(policy)
        style = _resolve_bar_style(policy, policy_index)
        values = [panel.transform(float(row[column])) for column in group_cols]
        offset = policy_index * width - (len(best) - 1) * width / 2.0
        positions = [x + offset for x in x_values]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=label,
            facecolor=style.get("facecolor", "white"),
            edgecolor=style.get("edgecolor", "black"),
            hatch=style.get("hatch"),
            linewidth=style.get("linewidth", 1.2),
            zorder=3,
        )
        if len(bars):
            _register_legend_entry(legend_handles, label, bars[0])

    ax.set_title(panel.title, fontsize=10)
    ax.set_xticks(x_values)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(panel.ylabel, fontsize=9)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#bfbfbf", zorder=0)
    ax.xaxis.grid(False)
    return True


def _render_bar_composites(
    df: pd.DataFrame, csv_path: Path, output_dir: Path
) -> list[Path]:
    output_paths: list[Path] = []

    for composite in BAR_COMPOSITES:
        panel_data: list[tuple[BarPanelSpec, pd.DataFrame]] = []
        panel_frames: list[pd.DataFrame] = []
        snapshot_scales: list[str] = []
        snapshot_stds: list[str] = []
        snapshot_scenarios: list[str] = []

        for panel in composite.panels:
            frame = _prepare_bar_frame_for_panel(df, panel)
            if frame is None:
                continue
            panel_data.append((panel, frame))
            panel_frames.append(frame)
            snapshot_scales.append(
                _sanitize_filename(
                    panel.snapshot_scale if panel.snapshot_scale is not None else DEFAULT_BAR_SCALE
                )
            )
            if panel.snapshot_scenario is not None:
                snapshot_scenarios.append(_sanitize_filename(panel.snapshot_scenario))
            else:
                snapshot_stds.append(
                    _sanitize_filename(
                        panel.snapshot_std if panel.snapshot_std is not None else DEFAULT_BAR_STD
                    )
                )

        if not panel_data:
            continue

        policies = _merge_policies(*panel_frames)
        if not policies:
            continue

        rows = math.ceil(len(panel_data) / composite.ncols)
        fig, axes_grid = plt.subplots(
            rows,
            composite.ncols,
            figsize=(5.0 * composite.ncols, 2.8 * rows),
            squeeze=False,
        )
        axes = axes_grid.flatten()
        legend_handles: dict[str, object] = {}

        for idx, (panel, frame) in enumerate(panel_data):
            ax = axes[idx]
            _plot_bar_panel(ax, frame, policies, panel, legend_handles)

        for unused_ax in axes[len(panel_data):]:
            unused_ax.set_visible(False)

        scale_suffix = snapshot_scales[0] if len(set(snapshot_scales)) == 1 else "mixed"
        if snapshot_scenarios:
            scenario_suffix = (
                snapshot_scenarios[0]
                if len(set(snapshot_scenarios)) == 1 and not snapshot_stds
                else "mixed"
            )
            output_name = (
                f"{csv_path.stem}_{composite.name}_scale_{scale_suffix}"
                f"_scenario_{scenario_suffix}.pdf"
            )
        else:
            std_suffix = snapshot_stds[0] if len(set(snapshot_stds)) == 1 else "mixed"
            output_name = (
                f"{csv_path.stem}_{composite.name}_scale_{scale_suffix}"
                f"_std_{std_suffix}.pdf"
            )
        output_path = output_dir / output_name
        _finalize_figure(
            fig,
            [ax for ax in axes[: len(panel_data)] if ax.get_visible()],
            output_path,
            legend_handles,
            ncols=len(policies),
        )
        output_paths.append(output_path)

    return output_paths


def _render_mixed_composites(
    df: pd.DataFrame, csv_path: Path, output_dir: Path
) -> list[Path]:
    if not MIXED_COMPOSITES:
        return []

    scale_metrics = _iter_mixed_line_metrics(MIXED_COMPOSITES, "scale_line")
    scale_prepared = _prepare_scale_frame(df, scale_metrics) if scale_metrics else None
    scale_frame = scale_prepared[0] if scale_prepared is not None else None
    std_metrics = _iter_mixed_line_metrics(MIXED_COMPOSITES, "std_line")
    std_frame = _prepare_std_frame(df, std_metrics) if std_metrics else None

    output_paths: list[Path] = []
    for composite in MIXED_COMPOSITES:
        bar_frames_by_index: dict[int, pd.DataFrame] = {}
        scale_frames_by_index: dict[int, tuple[str, pd.DataFrame] | None] = {}
        std_required = False
        for index, panel in enumerate(composite.panels):
            if panel.kind == "bar" and panel.bar is not None:
                frame = _prepare_bar_frame_for_panel(df, panel.bar)
                if frame is not None:
                    bar_frames_by_index[index] = frame
                continue
            if panel.kind == "scale_line" and panel.line is not None:
                scenario_name, frame = _select_scale_panel_frame(
                    scale_frame,
                    panel.line,
                    fallback_std=DEFAULT_BAR_STD,
                )
                if frame is not None and scenario_name is not None:
                    scale_frames_by_index[index] = (scenario_name, frame)
                continue
            if panel.kind == "std_line" and panel.line is not None:
                std_required = True

        policies = _merge_policies(scale_frame, std_frame, *bar_frames_by_index.values())
        if not policies:
            continue

        rows = math.ceil(len(composite.panels) / composite.ncols)
        fig, axes_grid = plt.subplots(
            rows,
            composite.ncols,
            figsize=(5.0 * composite.ncols, 2.8 * rows),
            squeeze=False,
        )
        axes = axes_grid.flatten()
        legend_handles: dict[str, object] = {}
        has_data = False
        used_scale_scenarios: list[str] = []

        for idx, panel in enumerate(composite.panels):
            ax = axes[idx]
            if panel.kind == "scale_line" and panel.line is not None:
                selected = scale_frames_by_index.get(idx)
                if selected is None:
                    _mark_no_data(
                        ax,
                        title=panel.line.title,
                        ylabel=panel.line.ylabel,
                        xlabel="API Traffic scale",
                        grid=False,
                    )
                else:
                    scenario_name, scenario_frame = selected
                    used_scale_scenarios.append(scenario_name)
                    has_data = _plot_scale_line_panel(
                        ax,
                        scenario_frame,
                        policies,
                        panel.line,
                        legend_handles,
                    ) or has_data
                continue

            if panel.kind == "std_line" and panel.line is not None:
                if std_frame is None or not std_required:
                    _mark_no_data(
                        ax,
                        title=panel.line.title,
                        ylabel=panel.line.ylabel,
                        xlabel="Instance Capacity Std",
                        grid=False,
                    )
                else:
                    has_data = _plot_std_line_panel(
                        ax,
                        std_frame,
                        policies,
                        panel.line,
                        legend_handles,
                    ) or has_data
                continue

            if panel.kind == "bar" and panel.bar is not None:
                bar_frame = bar_frames_by_index.get(idx)
                if bar_frame is None:
                    _mark_no_data(
                        ax,
                        title=panel.bar.title,
                        ylabel=panel.bar.ylabel,
                        grid=False,
                    )
                else:
                    has_data = _plot_bar_panel(
                        ax,
                        bar_frame,
                        policies,
                        panel.bar,
                        legend_handles,
                    ) or has_data
                continue

            _mark_no_data(ax, title="Unsupported panel", ylabel="", grid=False)

        for unused_ax in axes[len(composite.panels):]:
            unused_ax.set_visible(False)

        if not has_data:
            plt.close(fig)
            continue

        output_name = f"{csv_path.stem}_{composite.name}"
        if used_scale_scenarios:
            scale_suffix = (
                _sanitize_filename(used_scale_scenarios[0])
                if len(set(used_scale_scenarios)) == 1
                else "mixed_scenarios"
            )
            output_name = f"{output_name}_{scale_suffix}"
        output_path = output_dir / f"{output_name}.pdf"
        _finalize_figure(
            fig,
            [ax for ax in axes[: len(composite.panels)] if ax.get_visible()],
            output_path,
            legend_handles,
            ncols=max(2, len(policies)),
        )
        output_paths.append(output_path)

    return output_paths


def render_all(csv_path: Path, output_dir: Path | None = None) -> list[Path]:
    if not csv_path.exists() or not csv_path.is_file():
        raise ValueError(f"CSV file not found: {csv_path}")

    df = _load_csv(csv_path)
    output_dir = output_dir or csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    output_paths.extend(_render_line_composites_by_scale(df, csv_path, output_dir))
    output_paths.extend(_render_line_composites_by_std(df, csv_path, output_dir))
    output_paths.extend(_render_bar_composites(df, csv_path, output_dir))
    output_paths.extend(_render_mixed_composites(df, csv_path, output_dir))
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot sweep CSV figures from a single unified tool. "
            "The tool renders hard-coded line, grouped-bar, and mixed composite figures "
            "with shared legends by policy."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to result CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Defaults to the CSV directory.",
    )
    args = parser.parse_args()

    try:
        output_paths = render_all(args.csv, args.output_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if not output_paths:
        raise SystemExit("No figures were generated from the provided CSV.")

    for output_path in output_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
