import numpy as np


def ecdf(x):
    """
    Compute the empirical cumulative distribution function (ECDF).

    Parameters
    ----------
    x : array-like
        Sample observations.

    Returns
    -------
    xs : np.ndarray
        Sorted sample values.
    ys : np.ndarray
        ECDF values corresponding to xs.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must contain at least one observation")

    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys
