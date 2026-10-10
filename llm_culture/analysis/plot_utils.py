"""Shared matplotlib helpers for the analysis plot modules."""
import os
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pyplot's global current-figure/axes state is not thread-safe; the Flask dev
# server handles requests in threads, so serialize all plotting through this lock.
PLOT_LOCK = threading.RLock()


def setup_axes(title=None, xlabel=None, ylabel=None, sizes=None, pad=0, labelpad=0):
    """Set the title and axis labels using the configured font sizes."""
    sizes = sizes or {}
    if title is not None:
        plt.title(title, fontsize=sizes.get("title"), pad=pad)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=sizes.get("labels"), labelpad=labelpad)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=sizes.get("labels"), labelpad=labelpad)


def save_figure(fig_dir, filename, plot=False):
    """Write the current figure to ``fig_dir/filename`` (creating the dir), then close it."""
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, filename))
    print(f"Saved {filename}")
    if plot:
        plt.show()
    plt.close()
