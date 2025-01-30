import pytest
from sample_weight_audit.statistical_testing import ed_perm_test
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal
import numpy as np


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ed_perm_test(seed):
    rng = np.random.RandomState(seed)
    dim = 30
    means = rng.randn(dim)
    covariance = toeplitz(np.linspace(1, 0, dim), np.linspace(1, 0, dim))

    # a and b are equaly distributed:
    a = multivariate_normal(means, covariance).rvs(300, random_state=rng)
    b = multivariate_normal(means, covariance).rvs(300, random_state=rng)

    res = ed_perm_test(a, b, random_state=seed)
    assert res.pvalue > 0.05

    # modify the covariance matrix of b to make it different from a:
    covariance_b = covariance.copy()
    covariance_b[0, 0] *= 10
    b = multivariate_normal(means, covariance_b).rvs(300, random_state=rng)

    res = ed_perm_test(a, b, random_state=seed)
    assert res.pvalue < 0.05

    # modify the means of b to make it different from a:
    means_b = means.copy()
    means_b[0] += 10
    b = multivariate_normal(means_b, covariance).rvs(300, random_state=rng)

    res = ed_perm_test(a, b, random_state=seed)
    assert res.pvalue < 0.05