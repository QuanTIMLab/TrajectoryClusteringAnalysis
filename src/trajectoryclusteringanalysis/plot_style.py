"""Shared matplotlib style so that every figure is consistent and publication-ready."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter

PUBLICATION_RC = {
    # Fonts: one serif family for text and math, one size per role
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10,
    # Every title (axes or figure) shares the same font
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.titlesize': 13,
    'figure.titleweight': 'bold',
    # Clean look
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    # Export quality
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def set_publication_style():
    """Apply the publication style to all subsequent matplotlib figures."""
    plt.rcParams.update(PUBLICATION_RC)


def integer_ticks(axis):
    """Force integer tick values (epochs, time steps) on a matplotlib axis."""
    axis.set_major_locator(MaxNLocator(integer=True))


def float_ticks(axis, decimals=2):
    """Use the same fixed number of decimals for every tick of a matplotlib axis."""
    axis.set_major_formatter(
        FuncFormatter(lambda value, _: f'{value:.{decimals}f}'.replace('-', '\u2212'))
    )


def scientific_ticks(axis):
    """Scientific notation without an offset text, for axes whose magnitude is unknown."""
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((-2, 3))
    axis.set_major_formatter(formatter)
