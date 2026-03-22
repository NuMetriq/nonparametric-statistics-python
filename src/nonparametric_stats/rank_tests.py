import numpy as np
from scipy import stats


def sign_test(diffs):
    diffs = np.asarray(diffs)

    if diffs.ndim != 1:
        raise ValueError("diffs must be one-dimensional")
    if len(diffs) == 0:
        raise ValueError("diffs must contain at least one observation")

    diffs = diffs[diffs != 0]

    if len(diffs) == 0:
        raise ValueError("all differences are zero; sign test is undefined")

    n = len(diffs)
    n_pos = int(np.sum(diffs > 0))

    p_lower = stats.binom.cdf(n_pos, n, 0.5)
    p_upper = 1 - stats.binom.cdf(n_pos - 1, n, 0.5)
    p_value = min(1.0, 2 * min(p_lower, p_upper))

    return {
        "n": n,
        "n_pos": n_pos,
        "p_value": float(p_value),
    }


def mann_whitney_pairwise_less(x, y):
    """
    Count pairwise comparisons of x_i < y_j, with 0.5 credit for ties.

    This is useful for interpreting the Mann-Whitney framework as a
    pairwise probability P(X < Y), but it is not the same quantity as
    scipy.stats.mannwhitneyu(x, y).statistic, which returns the U
    statistic for the first sample x.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must both be non-empty")

    count = 0.0
    for xi in x:
        count += np.sum(xi < y)
        count += 0.5 * np.sum(xi == y)

    return float(count)
