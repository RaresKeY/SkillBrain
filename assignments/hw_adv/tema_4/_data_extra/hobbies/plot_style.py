# plot_style.py
import matplotlib.pyplot as plt

def apply_dark_theme():
    plt.rcParams.update({
        "axes.facecolor": "#1e1e1e", "figure.facecolor": "#1e1e1e",
        "axes.edgecolor": "#DDDDDD", "axes.labelcolor": "#DDDDDD",
        "xtick.color": "#CCCCCC", "ytick.color": "#CCCCCC",
        "text.color": "#DDDDDD", "grid.color": "#3a3a3a",
        "legend.edgecolor": "#1e1e1e", "legend.facecolor": "#2d2d30"
    })
