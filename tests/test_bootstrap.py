import numpy as np
from nonparametric_stats.bootstrap import (
    bootstrap_ci_basic,
    bootstrap_ci_normal,
    bootstrap_ci_percentile,
    bootstrap_distribution,
    bootstrap_resample,
)


def test_bootstrap_resample_length_matches_input():
    x = np.array([1, 2, 3, 4, 5])
    xb = bootstrap_resample(x, random_state=42)
    assert len(xb) == len(x)


def test_bootstrap_distribution_length():
    x = np.array([1, 2, 3, 4, 5])
    reps = bootstrap_distribution(np.mean, x, n_resamples=200, random_state=42)
    assert len(reps) == 200


def test_bootstrap_distribution_reproducible():
    x = np.array([1, 2, 3, 4, 5])
    reps1 = bootstrap_distribution(np.mean, x, n_resamples=200, random_state=42)
    reps2 = bootstrap_distribution(np.mean, x, n_resamples=200, random_state=42)
    assert np.allclose(reps1, reps2)


def test_bootstrap_percentile_ci_is_ordered():
    x = np.array([1, 2, 3, 4, 5])
    lower, upper = bootstrap_ci_percentile(np.mean, x, n_resamples=500, random_state=42)
    assert lower <= upper


def test_bootstrap_normal_ci_is_ordered():
    x = np.array([1, 2, 3, 4, 5])
    lower, upper = bootstrap_ci_normal(np.mean, x, n_resamples=500, random_state=42)
    assert lower <= upper


def test_bootstrap_basic_ci_is_ordered():
    x = np.array([1, 2, 3, 4, 5])
    lower, upper = bootstrap_ci_basic(np.mean, x, n_resamples=500, random_state=42)
    assert lower <= upper
