import numpy as np
from nonparametric_stats.permutation import (
    difference_in_means,
    difference_in_medians,
    permutation_test,
)


def test_difference_in_means_basic():
    x = [1, 2, 3]
    y = [4, 5, 6]
    assert difference_in_means(x, y) == -3.0


def test_difference_in_medians_basic():
    x = [1, 2, 100]
    y = [4, 5, 6]
    assert difference_in_medians(x, y) == -3.0


def test_permutation_test_returns_expected_keys():
    x = [1, 2, 3]
    y = [4, 5, 6]

    result = permutation_test(x, y, n_resamples=200, random_state=42)

    assert "observed_stat" in result
    assert "p_value" in result
    assert "permuted_stats" in result
    assert len(result["permuted_stats"]) == 200
    assert 0 <= result["p_value"] <= 1


def test_permutation_test_is_reproducible_with_seed():
    x = [1, 2, 3]
    y = [4, 5, 6]

    result1 = permutation_test(x, y, n_resamples=200, random_state=42)
    result2 = permutation_test(x, y, n_resamples=200, random_state=42)

    assert np.allclose(result1["permuted_stats"], result2["permuted_stats"])
    assert result1["p_value"] == result2["p_value"]
