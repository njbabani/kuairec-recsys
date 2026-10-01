import matplotlib as mpl
import matplotlib.pyplot as plt

from recsys.viz import style


def test_apply_style_sets_surface_ink_and_a_recessive_solid_grid():
    style.apply_style()

    assert mpl.rcParams["figure.facecolor"] == style.SURFACE
    assert mpl.rcParams["grid.color"] == style.GRID
    assert mpl.rcParams["grid.linestyle"] == "-"
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["lines.linewidth"] == style.LINE_WIDTH


def test_series_colors_come_from_the_validated_palette_in_fixed_order():
    style.apply_style()

    cycle = [entry["color"] for entry in mpl.rcParams["axes.prop_cycle"]]
    assert cycle == list(style.SERIES)


def test_add_figure_title_labels_the_whole_figure_without_touching_axes_titles():
    fig, (left, right) = plt.subplots(1, 2)

    style.add_figure_title(fig, "Exposure vs appeal", "Two panels")

    texts = {text.get_text(): text for text in fig.texts}
    assert texts["Exposure vs appeal"].get_color() == style.INK
    assert texts["Two panels"].get_color() == style.INK_SECONDARY
    assert left.get_title(loc="left") == right.get_title(loc="left") == ""
    plt.close(fig)


def test_add_title_sets_a_left_aligned_title_and_a_secondary_subtitle():
    fig, ax = plt.subplots()

    style.add_title(ax, "Viewers leave after ~7 seconds", "Median seconds watched")

    assert ax.get_title(loc="left") == "Viewers leave after ~7 seconds"
    subtitles = [text for text in ax.texts if text.get_text() == "Median seconds watched"]
    assert len(subtitles) == 1
    assert subtitles[0].get_color() == style.INK_SECONDARY
    plt.close(fig)
