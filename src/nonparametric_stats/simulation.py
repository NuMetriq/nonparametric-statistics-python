import numpy as np
from scipy import stats

from .permutation import (
    difference_in_means,
    difference_in_medians,
    permutation_test,
)


def welch_ttest_pvalue(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(stats.ttest_ind(x, y, equal_var=False).pvalue)


def mann_whitney_pvalue(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def permutation_mean_pvalue(x, y, n_resamples=2000, random_state=None):
    result = permutation_test(
        x,
        y,
        statistic_func=difference_in_means,
        n_resamples=n_resamples,
        alternative="two-sided",
        random_state=random_state,
    )
    return float(result["p_value"])


def permutation_median_pvalue(x, y, n_resamples=2000, random_state=None):
    result = permutation_test(
        x,
        y,
        statistic_func=difference_in_medians,
        n_resamples=n_resamples,
        alternative="two-sided",
        random_state=random_state,
    )
    return float(result["p_value"])


def simulate_rejection_rate(
    test_func,
    generator_x,
    generator_y,
    n=30,
    n_sim=500,
    alpha=0.05,
    seed=42,
):
    if n <= 0:
        raise ValueError("n must be positive")
    if n_sim <= 0:
        raise ValueError("n_sim must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    rng = np.random.default_rng(seed)
    rejections = 0

    for _ in range(n_sim):
        x = generator_x(rng, n)
        y = generator_y(rng, n)
        p = test_func(x, y)
        rejections += p < alpha

    return rejections / n_sim


def normal_generator(mean=0.0, std=1.0):
    def gen(rng, n):
        return rng.normal(loc=mean, scale=std, size=n)

    return gen


def laplace_generator(loc=0.0, scale=1.0):
    def gen(rng, n):
        return rng.laplace(loc=loc, scale=scale, size=n)

    return gen


def cauchy_generator(loc=0.0, scale=1.0):
    def gen(rng, n):
        return rng.standard_cauchy(size=n) * scale + loc

    return gen


def lognormal_generator(mean=0.0, sigma=1.0, shift=0.0):
    def gen(rng, n):
        return rng.lognormal(mean=mean, sigma=sigma, size=n) + shift

    return gen


def contaminated_normal_generator(mean=0.0, std=1.0, contam_std=8.0, contam_prob=0.1):
    def gen(rng, n):
        is_contam = rng.random(n) < contam_prob
        base = rng.normal(loc=mean, scale=std, size=n)
        contam = rng.normal(loc=mean, scale=contam_std, size=n)
        return np.where(is_contam, contam, base)

    return gen
