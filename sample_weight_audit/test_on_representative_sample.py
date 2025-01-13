import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, mannwhitneyu, ttest_ind
from scipy.sparse import csr_array


from .generate_weighted_and_repeated_data import get_estimator_dataset
from .multifit import multifit_over_weighted_and_repeated


def paired_test(
    est,
    test="kstest",
    threshold=0.05,
    correct_threshold=False,
    train_size=100,
    n_samples_per_cv_group=200,
    n_cv_group=3,
    n_features=10,
    n_classes=None,
    max_repeats=10,
    max_seed=200,
    rep_test_size=10,
    plot=False,
    issparse=False,
    **kwargs,
):
    """
    Note I assume predictions and predictions_ref are np.ndarray(predictions, samples)
    """

    start_time = time.time()

    X, y = get_estimator_dataset(est, n_samples_per_cv_group, n_cv_group, n_features, n_classes=n_classes)
    
    if issparse:
        X = csr_array(X)

    
    predictions_weighted, predictions_repeated, _ = multifit_over_weighted_and_repeated(
        est,
        X,
        y,
        max_seed=max_seed,
        rep_test_size=rep_test_size,
        train_size=train_size,
        n_cv_group=n_cv_group,
        max_repeats=max_repeats,
        n_classes=n_classes,
    )
    # print(X.shape, y.shape)
    diffs = predictions_weighted.max(axis=0) - predictions_weighted.min(axis=0)

    if np.all(diffs < np.finfo(diffs.dtype).eps):
        raise ValueError(
            f"repeatedly fitting {est} with different random state led to the same predictions"
        )

    diffs = predictions_repeated.max(axis=0) - predictions_repeated.min(axis=0)

    if np.all(diffs < np.finfo(diffs.dtype).eps):

        raise ValueError(
            f"repeatedly fitting {est} with different random state led to the same predictions"
        )

    if plot:
        fig, axs = plt.subplots(
            2, int(predictions_weighted.shape[1] / 2), figsize=(12, 6)
        )
        i = 0
        for ax in axs.flatten():
            ax.hist(predictions_repeated[:, i], label="repeated", bins=10, density=True)
            ax.hist(
                predictions_weighted[:, i],
                alpha=0.7,
                label="weighted",
                bins=10,
                density=True,
            )
            if i == (predictions_weighted.shape[1] - 1):
                plt.legend()
            i += 1

        plt.show()
    if correct_threshold:
        threshold /= predictions_repeated.shape[1]

    p_vals = []
    test_statistic = []

    for pred, pred_ref in zip(predictions_weighted.T, predictions_repeated.T):
        if test == "kstest":
            test_result = kstest(pred, pred_ref, **kwargs)
        elif test == "welch":
            test_result = ttest_ind(
                pred, pred_ref, **kwargs
            )  # hard code equal_var = False
        elif test == "mannwhitneyu":
            test_result = mannwhitneyu(pred, pred_ref, **kwargs)

        p_vals.append(test_result.pvalue)
        test_statistic.append(test_result.statistic)

    print(
        "Finished looping till the maximum random state,",
        max_seed,
        "for estimator",
        est,
        "in ----",
        time.time() - start_time,
        "s---",
    )
    print("Minimum p-values: ", np.array(p_vals).min())

    return {
        "Name": est.__name__,
        "p_values": p_vals,
        "min_p_value": np.array(p_vals).min(),
        "avg_p_value": np.nanmean(p_vals),
    }