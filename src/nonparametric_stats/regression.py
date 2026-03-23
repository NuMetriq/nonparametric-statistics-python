import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.nonparametric.smoothers_lowess import lowess


def validate_1d_regression_inputs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) == 0:
        raise ValueError("x and y must be non-empty")

    return x, y


def fit_linear_regression_polyfit(x, y, x_grid):
    x, y = validate_1d_regression_inputs(x, y)
    x_grid = np.asarray(x_grid, dtype=float)

    coef = np.polyfit(x, y, deg=1)
    y_hat = np.polyval(coef, x_grid)

    return {
        "coef": coef,
        "x_grid": x_grid,
        "y_hat": y_hat,
    }


def fit_lowess(x, y, frac=0.2):
    x, y = validate_1d_regression_inputs(x, y)

    if not (0 < frac <= 1):
        raise ValueError("frac must be in (0, 1]")

    fitted = lowess(endog=y, exog=x, frac=frac, return_sorted=True)

    return {
        "x_sorted": fitted[:, 0],
        "y_hat": fitted[:, 1],
    }


def fit_knn_regression(x, y, x_grid, n_neighbors=10):
    x, y = validate_1d_regression_inputs(x, y)
    x_grid = np.asarray(x_grid, dtype=float)

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(x.reshape(-1, 1), y)
    y_hat = model.predict(x_grid.reshape(-1, 1))

    return {
        "x_grid": x_grid,
        "y_hat": y_hat,
    }


def fit_kernel_regression(x, y, x_grid, bandwidth=None):
    x, y = validate_1d_regression_inputs(x, y)
    x_grid = np.asarray(x_grid, dtype=float)

    if bandwidth is None:
        kr = KernelReg(endog=y, exog=x, var_type="c")
    else:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        kr = KernelReg(endog=y, exog=x, var_type="c", bw=[bandwidth])

    y_hat, _ = kr.fit(x_grid)

    return {
        "x_grid": x_grid,
        "y_hat": y_hat,
    }


def mean_squared_error_manual(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    return float(np.mean((y_true - y_pred) ** 2))
