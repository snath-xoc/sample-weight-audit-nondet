from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.stats import kstest, mannwhitneyu, ttest_ind
from scipy.spatial.distance import cdist

from sklearn.utils import check_random_state


def energy_distance(x, y):
    n, m = len(x), len(y)
    D_xx = cdist(x, x)
    D_yy = cdist(y, y)
    D_xy = cdist(x, y)
    return (
        2 * D_xy.sum() / (n * m)
        - D_xx.sum() / (n * (n - 1))
        - D_yy.sum() / (m * (m - 1))
    ) / 2


@dataclass
class EnerygDistancePermutationTestResult:
    statistic: List[np.ndarray]
    pvalue: float


def ed_perm_test(x, y, n_perm=100, random_state=None):
    rng = check_random_state(random_state)
    assert x.ndim == 2  # (sample_size, obs_dim)
    assert x.shape == y.shape

    ed_obs = energy_distance(x, y)
    n, _ = len(x), len(y)
    z = np.vstack([x, y])
    ed_perm = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(z)
        ed_perm[i] = energy_distance(z[:n], z[n:])

    pvalue = ((ed_perm >= ed_obs).sum() + 1) / (n_perm + 1)
    return EnerygDistancePermutationTestResult([ed_perm, ed_obs], pvalue)


def run_1d_test(pred_ref, pred, test_name="kstest"):
    if test_name == "kstest":
        test_result = kstest(pred, pred_ref)
    elif test_name == "welch":
        test_result = ttest_ind(pred, pred_ref, equal_var=False)
    elif test_name == "mannwhitneyu":
        test_result = mannwhitneyu(pred, pred_ref)
    else:
        raise ValueError(
            f"Test {test_name} is not supported, please use 'kstest', 'welch' or 'mannwhitneyu'"
        )
    return test_result
