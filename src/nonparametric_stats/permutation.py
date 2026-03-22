import numpy as np


def difference_in_means(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must both be non-empty")

    return float(np.mean(x) - np.mean(y))


def difference_in_medians(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must both be non-empty")

    return float(np.median(x) - np.median(y))


def permutation_test(
    x,
    y,
    statistic_func=difference_in_means,
    n_resamples=5000,
    alternative="two-sided",
    random_state=None,
):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x and y must both be non-empty")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    rng = np.random.default_rng(random_state)

    observed_stat = statistic_func(x, y)
    combined = np.concatenate([x, y])
    n_x = len(x)

    permuted_stats = np.empty(n_resamples)

    for b in range(n_resamples):
        permuted = rng.permutation(combined)
        x_star = permuted[:n_x]
        y_star = permuted[n_x:]
        permuted_stats[b] = statistic_func(x_star, y_star)

    if alternative == "two-sided":
        p_value = np.mean(np.abs(permuted_stats) >= abs(observed_stat))
    elif alternative == "greater":
        p_value = np.mean(permuted_stats >= observed_stat)
    else:
        p_value = np.mean(permuted_stats <= observed_stat)

    return {
        "observed_stat": float(observed_stat),
        "p_value": float(p_value),
        "permuted_stats": permuted_stats,
    }
