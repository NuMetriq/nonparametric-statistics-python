import numpy as np


def gaussian_kernel(u):
    """
    Standard Gaussian kernel evaluated at u.
    """
    u = np.asarray(u, dtype=float)
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)


def gaussian_kde_manual(x, grid, bandwidth):
    """
    Manual Gaussian kernel density estimator.

    Parameters
    ----------
    x : array-like
        One-dimensional sample.
    grid : array-like
        Points at which to evaluate the density estimate.
    bandwidth : float
        Positive bandwidth parameter.

    Returns
    -------
    np.ndarray
        KDE values evaluated on the grid.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if grid.ndim != 1:
        raise ValueError("grid must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    vals = np.empty(len(grid), dtype=float)
    const = 1.0 / (len(x) * bandwidth)

    for i, g in enumerate(grid):
        vals[i] = const * np.sum(gaussian_kernel((g - x) / bandwidth))

    return vals


def silverman_bandwidth(x):
    """
    Silverman's rule-of-thumb bandwidth for 1D data.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) < 2:
        raise ValueError("x must contain at least two observations")

    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std

    if sigma <= 0:
        raise ValueError("data must have positive spread")

    n = len(x)
    return 0.9 * sigma * n ** (-1 / 5)


def approximate_integral(grid, values):
    """
    Approximate integral using the trapezoid rule.
    """
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)

    if grid.ndim != 1 or values.ndim != 1:
        raise ValueError("grid and values must be one-dimensional")
    if len(grid) != len(values):
        raise ValueError("grid and values must have the same length")
    if len(grid) < 2:
        raise ValueError("grid must contain at least two points")

    return float(np.trapezoid(values, grid))
