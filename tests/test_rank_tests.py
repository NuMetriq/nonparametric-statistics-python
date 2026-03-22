import math

from nonparametric_stats.rank_tests import mann_whitney_pairwise_less, sign_test
from scipy import stats


def test_sign_test_basic():
    diffs = [1, 2, 3, -1]
    result = sign_test(diffs)

    assert result["n"] == 4
    assert result["n_pos"] == 3
    assert 0 <= result["p_value"] <= 1


def test_mann_whitney_pairwise_less_basic():
    x = [1, 2]
    y = [3, 4]

    u = mann_whitney_pairwise_less(x, y)
    assert math.isclose(u, 4.0)


def test_mann_whitney_pairwise_less_complements_scipy_u():
    x = [1, 2, 3]
    y = [4, 5]

    pairwise_less = mann_whitney_pairwise_less(x, y)
    u_scipy = stats.mannwhitneyu(x, y, alternative="two-sided").statistic

    assert pairwise_less + u_scipy == len(x) * len(y)
