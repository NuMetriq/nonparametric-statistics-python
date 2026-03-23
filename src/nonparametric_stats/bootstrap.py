import numpy as np


def bootstrap_resample(x, random_state=None):
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must be non-empty")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(x), size=len(x))
    return x[indices]


def bootstrap_distribution(stat_func, x, n_resamples=5000, random_state=None):
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(random_state)
    reps = np.empty(n_resamples)

    for b in range(n_resamples):
        sample_idx = rng.integers(0, len(x), size=len(x))
        xb = x[sample_idx]
        reps[b] = stat_func(xb)

    return reps


def bootstrap_ci_percentile(
    stat_func, x, alpha=0.05, n_resamples=5000, random_state=None
):
    reps = bootstrap_distribution(
        stat_func=stat_func,
        x=x,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    lower = np.quantile(reps, alpha / 2)
    upper = np.quantile(reps, 1 - alpha / 2)
    return float(lower), float(upper)


def bootstrap_ci_normal(stat_func, x, alpha=0.05, n_resamples=5000, random_state=None):
    from scipy import stats

    x = np.asarray(x)
    theta_hat = stat_func(x)
    reps = bootstrap_distribution(
        stat_func=stat_func,
        x=x,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    se_boot = np.std(reps, ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)
    lower = theta_hat - z * se_boot
    upper = theta_hat + z * se_boot
    return float(lower), float(upper)


def bootstrap_ci_basic(stat_func, x, alpha=0.05, n_resamples=5000, random_state=None):
    x = np.asarray(x)
    theta_hat = stat_func(x)
    reps = bootstrap_distribution(
        stat_func=stat_func,
        x=x,
        n_resamples=n_resamples,
        random_state=random_state,
    )
    q_low = np.quantile(reps, alpha / 2)
    q_high = np.quantile(reps, 1 - alpha / 2)
    lower = 2 * theta_hat - q_high
    upper = 2 * theta_hat - q_low
    return float(lower), float(upper)
