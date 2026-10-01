import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def isolated_matplotlib():
    """apply_style() mutates global rcParams; restore them and close figures after each test."""
    with mpl.rc_context():
        yield
    plt.close("all")
