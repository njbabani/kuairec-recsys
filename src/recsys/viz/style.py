"""House style for static figures (EDA notebook, README): one validated palette, quiet chrome.

The two series colors pass the colorblind-safety and contrast checks on the light surface
(adjacent CVD ΔE 24.7, normal-vision ΔE 33.6). Static PNGs can't follow a dark theme, so
figures always render on the light surface.
"""

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")  # fixed order: blue, then orange
PRIMARY, SECONDARY = SERIES
AREA_ALPHA = 0.10
LINE_WIDTH = 2.0
FIGURE_DPI = 120
SUBTITLE_SIZE = 9.5
SUBTITLE_OFFSET_PT = 18
FIGURE_TITLE_LEFT = 0.01
FIGURE_TITLE_TOP = 0.88  # top of the axes area when a figure-level title is used


def apply_style() -> None:
    """Set matplotlib defaults: light surface, hairline solid y-grid, thin marks, no box."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
            "axes.titlelocation": "left",
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.titlecolor": INK,
            "axes.titlepad": 20,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "ytick.left": False,
            "lines.linewidth": LINE_WIDTH,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "legend.frameon": False,
        }
    )


def add_figure_title(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title for a multi-panel figure.

    Figure-level text is ignored by ``tight_layout``, so a long title can't push panels apart;
    call ``fig.tight_layout(rect=(0, 0, 1, FIGURE_TITLE_TOP))`` to leave room for it.
    """
    fig.text(
        FIGURE_TITLE_LEFT,
        0.99,
        title,
        ha="left",
        va="top",
        color=INK,
        fontsize=mpl.rcParams["axes.titlesize"],
        fontweight=mpl.rcParams["axes.titleweight"],
    )
    if subtitle:
        below_title = offset_copy(fig.transFigure, fig=fig, y=-SUBTITLE_OFFSET_PT, units="points")
        fig.text(
            FIGURE_TITLE_LEFT,
            0.99,
            subtitle,
            ha="left",
            va="top",
            color=INK_SECONDARY,
            fontsize=SUBTITLE_SIZE,
            transform=below_title,
        )


def add_title(ax: Axes, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title in primary ink, optional one-line subtitle in secondary ink."""
    ax.set_title(title, loc="left")
    if subtitle:
        ax.text(
            0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=SUBTITLE_SIZE,
            va="bottom",
        )
