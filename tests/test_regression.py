import numpy as np
from nonparametric_stats.regression import (
    fit_kernel_regression,
    fit_knn_regression,
    fit_linear_regression_polyfit,
    fit_lowess,
    mean_squared_error_manual,
    validate_1d_regression_inputs,
)


def test_validate_1d_regression_inputs_basic():
    x = [1, 2, 3]
    y = [4, 5, 6]
    x_out, y_out = validate_1d_regression_inputs(x, y)
    assert len(x_out) == 3
    assert len(y_out) == 3


def test_linear_regression_returns_grid_length():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    grid = np.linspace(0, 3, 20)

    result = fit_linear_regression_polyfit(x, y, grid)
    assert len(result["x_grid"]) == 20
    assert len(result["y_hat"]) == 20


def test_lowess_returns_sorted_output():
    x = np.array([3.0, 1.0, 2.0, 0.0])
    y = np.array([3.0, 1.0, 2.0, 0.0])

    result = fit_lowess(x, y, frac=0.5)
    assert len(result["x_sorted"]) == 4
    assert np.all(np.diff(result["x_sorted"]) >= 0)


def test_knn_regression_returns_grid_length():
    x = np.linspace(0, 1, 10)
    y = x**2
    grid = np.linspace(0, 1, 25)

    result = fit_knn_regression(x, y, grid, n_neighbors=3)
    assert len(result["x_grid"]) == 25
    assert len(result["y_hat"]) == 25


def test_kernel_regression_returns_grid_length():
    x = np.linspace(0, 1, 15)
    y = np.sin(x)
    grid = np.linspace(0, 1, 30)

    result = fit_kernel_regression(x, y, grid, bandwidth=0.2)
    assert len(result["x_grid"]) == 30
    assert len(result["y_hat"]) == 30


def test_mse_manual_zero_when_equal():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error_manual(y_true, y_pred) == 0.0
