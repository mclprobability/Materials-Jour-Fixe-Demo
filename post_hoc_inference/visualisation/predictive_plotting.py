import numpy as np
from matplotlib import pyplot as plt


def plot_1d_predictive_density(x: np.ndarray, y_pred: np.ndarray, ci=0.9, label="", ax=None):
    """Plot the result of a 1d function mapping from R^1 --> R^1
    Args:
        ax (): Matplotlib Axis object
        x (np.ndarray): shape (N)
        y_inf (np.ndarray): Result of Inference evaluated at x. shape(B, N)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    y_pred_mean = y_pred.mean(axis=0)
    ci_low = np.quantile(y_pred, axis=0, q=0.5 - ci / 2.0)
    ci_high = np.quantile(y_pred, axis=0, q=0.5 + ci / 2.0)
    ax.plot(x, y_pred_mean, label=f"{label}", alpha=1.0, zorder=3)
    ax.fill_between(x, ci_low, ci_high, label=f"{label} {ci} % CI", alpha=0.2, zorder=1)
    ax.legend()
    return ax
