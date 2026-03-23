import numpy as np
from nonparametric_stats.kde import (
    approximate_integral,
    gaussian_kde_manual,
    gaussian_kernel,
    silverman_bandwidth,
)


def test_gaussian_kernel_at_zero():
    val = gaussian_kernel(0.0)
    expected = 1 / np.sqrt(2 * np.pi)
    assert np.isclose(val, expected)


def test_gaussian_kde_manual_returns_correct_length():
    x = np.array([0.0, 1.0, 2.0])
    grid = np.linspace(-1, 3, 50)
    vals = gaussian_kde_manual(x, grid, bandwidth=0.5)
    assert len(vals) == len(grid)


def test_gaussian_kde_manual_nonnegative():
    x = np.array([0.0, 1.0, 2.0])
    grid = np.linspace(-1, 3, 50)
    vals = gaussian_kde_manual(x, grid, bandwidth=0.5)
    assert np.all(vals >= 0)


def test_silverman_bandwidth_positive():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bw = silverman_bandwidth(x)
    assert bw > 0


def test_approximate_integral_reasonable_for_kde():
    x = np.array([-1.0, 0.0, 1.0])
    grid = np.linspace(-5, 5, 2000)
    vals = gaussian_kde_manual(x, grid, bandwidth=0.5)
    integral = approximate_integral(grid, vals)
    assert 0.98 <= integral <= 1.02
