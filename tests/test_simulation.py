from nonparametric_stats.simulation import (
    mann_whitney_pvalue,
    normal_generator,
    permutation_mean_pvalue,
    permutation_median_pvalue,
    simulate_rejection_rate,
    welch_ttest_pvalue,
)


def test_welch_ttest_pvalue_in_unit_interval():
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = welch_ttest_pvalue(x, y)
    assert 0 <= p <= 1


def test_mann_whitney_pvalue_in_unit_interval():
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = mann_whitney_pvalue(x, y)
    assert 0 <= p <= 1


def test_permutation_mean_pvalue_in_unit_interval():
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = permutation_mean_pvalue(x, y, n_resamples=200, random_state=42)
    assert 0 <= p <= 1


def test_permutation_median_pvalue_in_unit_interval():
    x = [1, 2, 3]
    y = [4, 5, 6]
    p = permutation_median_pvalue(x, y, n_resamples=200, random_state=42)
    assert 0 <= p <= 1


def test_simulate_rejection_rate_in_unit_interval():
    gen0 = normal_generator(mean=0.0, std=1.0)
    rate = simulate_rejection_rate(
        test_func=welch_ttest_pvalue,
        generator_x=gen0,
        generator_y=gen0,
        n=20,
        n_sim=50,
        alpha=0.05,
        seed=42,
    )
    assert 0 <= rate <= 1
