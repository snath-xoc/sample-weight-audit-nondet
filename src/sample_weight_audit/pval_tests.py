import numpy as np
from scipy.stats import kruskal, kstest, mannwhitneyu, ttest_ind
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


def kruskal_pval(x, y, verbose=False, **kwargs):
    test_results = kruskal(*(x.flatten() - y.flatten()).T)
    if verbose:
        print(f"kruskal diff p-value: {test_results.pvalue:.4f}")
    return test_results


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


# energy_distance(a, b)
# Permutation test to check that two nd arrays have the same distribution
# using the energy distance:


def ed_perm_test(x, y, n_perm=100, random_seed=None, verbose=False, **kwargs):
    C = type("C", (object,), {})
    test_results = C()
    rng = np.random.default_rng(random_seed)
    ed_obs = energy_distance(x, y)
    n, _ = len(x), len(y)
    z = np.vstack([x, y])
    ed_perm = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(z)
        ed_perm[i] = energy_distance(z[:n], z[n:])

    test_results.statistic = [ed_perm, ed_obs]
    test_results.pvalue = ((ed_perm >= ed_obs).sum() + 1) / (n_perm + 1)
    if verbose:
        print(f"ed perm p-value: {test_results.pvalue:.3f}")
    return test_results


def get_pval(pred_ref, pred, test="kstest", **kwargs):

    if test == "kstest":
        test_result = kstest(pred, pred_ref)
    elif test == "welch":
        kwargs_ttest_ind = {}
        if "equal_var" in kwargs:
            kwargs_ttest_ind["equal_var"] = kwargs["equal_var"]
        test_result = ttest_ind(
            pred, pred_ref, **kwargs_ttest_ind
        )  # hard code equal_var = False
    elif test == "mannwhitneyu":
        test_result = mannwhitneyu(pred, pred_ref)

    elif test == "ed_perm":
        test_result = ed_perm_test(pred, pred_ref, **kwargs)

    else:
        print(f"{test} not implemented yet!")
    return test_result


def scan_for_pvalue(
    preds, preds_ref, test="kstest", sample_mult_dim=60, projection_type="PCA", **kwargs
):
    """
    Function to scan over output dimensions with minimum pvalue
    """
    assert len(preds.shape) == len(preds_ref.shape)
    assert projection_type in ["GaussianRandomProjection", "PCA"]

    if len(preds.shape) > 1 and test != "ed_perm":

        if preds.shape[0] * preds.shape[1] != sample_mult_dim:
            n_components = int(sample_mult_dim // preds.shape[0])
            if projection_type == "PCA":
                preds = PCA.fit_transform(preds, n_components=n_components)
                preds_ref = PCA.fit_transform(preds_ref, n_components=n_components)
            elif projection_type == "GaussianRandomProjection":
                components = (
                    GaussianRandomProjection(n_components=n_components)
                    .fit(preds)
                    .components_
                )
                preds = preds @ components.T
                preds_ref = preds_ref @ components.T

        test_results = []
        pvals = []
        for pred, pred_ref in zip(preds.T, preds_ref.T):
            test_results.append(get_pval(pred_ref, pred, test=test, **kwargs))
            pvals.append(test_results[-1].pvalue)
        min_p_val_idx = np.argmin(np.asarray(pvals))
        test_results = test_results[min_p_val_idx]
        preds_ref_plot = preds_ref[:, min_p_val_idx]
        preds_plot = preds[:, min_p_val_idx]
    else:
        test_results = get_pval(preds_ref, preds, test=test, **kwargs)
        preds_ref_plot = preds_ref
        preds_plot = preds
    return test_results, preds_plot, preds_ref_plot
