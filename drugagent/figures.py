"""Figures for the README and the talk: one quiet style, colours from one validated palette.

Diverging data (down / up regulation) uses the blue <-> red pair with gray for
"nothing"; categorical series take slots in a fixed order. Text is never coloured
by series. Every figure is saved as PNG (300 dpi) and SVG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, in order
DOWN, UP, NEUTRAL = "#2a78d6", "#e34948", "#c9c8c2"


def style(ax, title: str | None = None, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10, fontweight="bold")


def figure(width: float = 7.0, height: float = 4.2, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)
    return fig, axes


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300 if ext == "png" else None, facecolor=SURFACE)
    plt.close(fig)
    return FIGURES / f"{name}.png"


OFFSETS = [(5, 2, "left"), (5, -11, "left"), (-5, 2, "right"), (-5, -11, "right"), (5, 13, "left"),
           (-5, 13, "right"), (14, -4, "left"), (-14, -4, "right"), (5, 24, "left"), (-5, 24, "right"),
           (5, -22, "left"), (-5, -22, "right")]


def label_points(ax, points: list[tuple[float, float, str]], fontsize: float = 7.5) -> None:
    """Direct labels: the first offset around the point whose box is inside the axes and clear of the rest."""
    fig = ax.figure
    fig.tight_layout()  # fix the axes' final position first; save() re-running it is then a no-op
    fig.canvas.draw()
    renderer, placed = fig.canvas.get_renderer(), []
    frame = ax.get_window_extent(renderer)
    for x, y, text in sorted(points, key=lambda p: -p[1]):
        for k, (dx, dy, align) in enumerate(OFFSETS):
            label = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                                color=INK, va="bottom", ha=align)
            box = label.get_window_extent(renderer).expanded(1.05, 1.15)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            if inside and not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            label.remove()  # no clear spot: better no label than one on top of another


def volcano(result, name: str, title: str, label_top: int = 8) -> Path:
    """log2 fold change against -log10 adjusted p; significant genes coloured by direction."""
    tested = result.tested
    lfc, padj = result.log2fc[tested], result.padj[tested]
    y = -np.log10(np.maximum(padj, 1e-300))
    genes = np.array(result.genes)[tested]
    sig = padj < result.fdr
    fig, ax = figure(6.4, 4.6)
    style(ax, title, "log2 fold change (treated / control)", "-log10 adjusted p")
    ax.scatter(lfc[~sig], y[~sig], s=9, color=NEUTRAL, linewidths=0, label="not significant")
    down, up = sig & (lfc < 0), sig & (lfc > 0)
    ax.scatter(lfc[down], y[down], s=14, color=DOWN, edgecolors=SURFACE, linewidths=0.5, label=f"down ({down.sum()})")
    ax.scatter(lfc[up], y[up], s=14, color=UP, edgecolors=SURFACE, linewidths=0.5, label=f"up ({up.sum()})")
    ax.axhline(-np.log10(result.fdr), color=INK_3, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(ax.get_xlim()[0], -np.log10(result.fdr), f" FDR {result.fdr:g}", color=INK_2, fontsize=8, va="bottom")
    label_points(ax, [(lfc[k], y[k], genes[k]) for k in np.argsort(padj)[:label_top]])
    legend = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_2)
    return save(fig, name)
