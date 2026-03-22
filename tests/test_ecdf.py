import numpy as np
from nonparametric_stats.ecdf import ecdf


def test_ecdf_sorts_values():
    x = np.array([3, 1, 2])
    xs, ys = ecdf(x)

    assert np.array_equal(xs, np.array([1, 2, 3]))
    assert np.allclose(ys, np.array([1 / 3, 2 / 3, 1.0]))


def test_ecdf_raises_on_empty():
    try:
        ecdf([])
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_ecdf_last_value_is_one():
    xs, ys = ecdf([10, 20, 30, 40])
    assert ys[-1] == 1.0
