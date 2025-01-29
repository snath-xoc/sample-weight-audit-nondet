from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.random_projection import GaussianRandomProjection
from tqdm import tqdm

from .data import (
    get_diverse_subset,
    make_data_for_estimator,
    make_weighted_and_repeated_train_test,
)
from .estimator_params import STOCHASTIC_FIT_PARAMS
from .statistical_testing import ed_perm_test, run_1d_test


@dataclass
class EquivalenceTestResult:
    name: str
    min_p_value: float
    mean_p_value: float
    p_values: np.ndarray
    predictions_weighted: np.ndarray
    predictions_repeated: np.ndarray

    def __repr__(self):
        return (
            f"EquivalenceTestResult(name={self.name}, min_p_value={self.min_p_value}, "
            f"mean_p_value={self.mean_p_value})"
        )


def check_weighted_repeated_estimator_fit_equivalence(
    est,
    test_name="ed_perm",
    n_samples_per_cv_group=200,
    n_cv_group=3,
    n_features=10,
    n_classes=3,
    max_sample_weight=10,
    n_stochastic_fits=200,
    stat_test_dim=30,
    fit_on_sparse_data=False,
    random_state=None,
):
    """Assess the correct use of weights for estimators with stochastic fits.

    This function fits an estimator with different random seeds, either with
    weighted data or repeated data. It then runs a statistical test to check if
    the predictions are equivalently distributed.

    The statistical test supports n-dimensional predictions and returns a
    single p-value by scanning through all test dimensions and returning the
    minimum p-value. Note that we enforce a common dimension for the
    statistical test to make it possible to somewhat meaningfully compare the
    p-values across different estimators.

    The test dimensionality `stat_test_dim` is the product of the number of
    test data points for which predictions are computed and the dimensionality
    of the predictions (1 for regressors, n_classes for classifiers, and nd for
    transformers).

    """

    X, y = make_data_for_estimator(
        est, n_samples_per_cv_group * n_cv_group, n_features, n_classes=n_classes
    )

    if fit_on_sparse_data:
        X = csr_array(X)

    predictions_weighted, predictions_repeated, _ = multifit_over_weighted_and_repeated(
        est,
        X,
        y,
        n_stochastic_fits=n_stochastic_fits,
        stat_test_dim=stat_test_dim,
        n_samples_per_cv_group=n_samples_per_cv_group,
        n_cv_group=n_cv_group,
        max_sample_weight=max_sample_weight,
        random_state=random_state,
    )
    assert (
        predictions_weighted.ndim == 3
    )  # (n_stochastic_fits, n_test_data_points, prediction_dim)
    assert predictions_weighted.shape == predictions_repeated.shape

    message = (
        f"Repeatedly fitting {est} with different random seeds led to the "
        "same predictions: please check sample weight equivalence by an exact "
        " equality test instead of statistical tests."
    )
    diffs = predictions_weighted.max(axis=0) - predictions_weighted.min(axis=0)
    if np.all(diffs < np.finfo(diffs.dtype).eps):
        raise ValueError(message)

    diffs = predictions_repeated.max(axis=0) - predictions_repeated.min(axis=0)
    if np.all(diffs < np.finfo(diffs.dtype).eps):
        raise ValueError(message)

    p_vals = []

    data_to_test_weighted = predictions_weighted.reshape(
        n_stochastic_fits, stat_test_dim
    ).T
    data_to_test_repeated = predictions_repeated.reshape(
        n_stochastic_fits, stat_test_dim
    ).T

    if test_name == "ed_perm":
        test_results = ed_perm_test(
            data_to_test_weighted, data_to_test_repeated, random_state=random_state
        )
        p_vals = [test_results.pvalue]
    else:
        # Iterate of all statistical test dimensions and compute p-values
        # for each dimension.
        for i in range(stat_test_dim):
            pvalue = run_1d_test(
                data_to_test_weighted[i], data_to_test_repeated[i], test_name
            ).pvalue
            p_vals.append(pvalue)

    p_vals = np.asarray(p_vals)
    return EquivalenceTestResult(
        est.__class__.__name__,
        p_vals.min(),
        np.nanmean(p_vals),
        p_vals,
        predictions_weighted,
        predictions_repeated,
    )


def non_default_params(est):
    est_class = est.__class__
    est_name = est_class.__module__ + "." + est_class.__name__
    return STOCHASTIC_FIT_PARAMS.get(est_name, {})


def add_cv_params(
    extra_params,
    n_cv_group,
    n_samples_per_cv_group,
    X_train,
    X_resampled_by_weights,
    sample_weight,
):
    extra_params_weighted = extra_params.copy()
    groups_weighted = np.hstack(
        ([np.full(n_samples_per_cv_group, i) for i in range(n_cv_group)])
    )
    splits_weighted = list(LeaveOneGroupOut().split(X_train, groups=groups_weighted))
    extra_params_weighted["cv"] = splits_weighted

    extra_params_repeated = extra_params.copy()
    groups_repeated = np.repeat(groups_weighted, sample_weight.astype(int), axis=0)
    splits_repeated = list(
        LeaveOneGroupOut().split(X_resampled_by_weights, groups=groups_repeated)
    )
    extra_params_repeated["cv"] = splits_repeated
    return extra_params_weighted, extra_params_repeated


def compute_predictions(est, X):
    if is_regressor(est):
        # Reshape to 2D to match classifier and tranformer output shapes.
        return est.predict(X).reshape(-1, 1)
    elif is_classifier(est):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)
        else:
            return est.decision_function(X)
    elif hasattr(est, "transform"):
        return est.transform(X)
    else:
        raise NotImplementedError(f"Estimator type not supported: {est}")


def multifit_over_weighted_and_repeated(
    est,
    X,
    y,
    n_stochastic_fits=200,
    stat_test_dim=30,
    n_samples_per_cv_group=300,
    n_cv_group=3,
    max_sample_weight=10,
    random_state=None,
):
    extra_params = non_default_params(est)

    if "cv" in est.get_params():
        use_cv = True
        effective_train_size = n_samples_per_cv_group * n_cv_group
    else:
        use_cv = False
        effective_train_size = n_samples_per_cv_group

    (
        X_train,
        y_train,
        X_resampled_by_weights,
        y_resampled_by_weights,
        sample_weight,
        X_test,
        _,
    ) = make_weighted_and_repeated_train_test(
        X,
        y,
        train_size=effective_train_size,
        max_sample_weight=max_sample_weight,
        random_state=random_state,
    )

    if use_cv:
        extra_params_weighted, extra_params_repeated = add_cv_params(
            extra_params,
            n_cv_group,
            n_samples_per_cv_group,
            X_train,
            X_resampled_by_weights,
            sample_weight,
        )
    else:
        extra_params_weighted = extra_params.copy()
        extra_params_repeated = extra_params.copy()

    # Perform one reference fit to inspect the predictions dimensions.
    est_ref = clone(est).set_params(random_state=0, **extra_params_weighted)
    est_ref.fit(X_train, y_train, sample_weight=sample_weight)

    predictions_ref = compute_predictions(est_ref, X_test)

    # Adjust the number of predictions so that stat_test_dim = n_test_data_points *
    # prediction_dim for all evaluated models. This is necessary to
    assert predictions_ref.ndim == 2
    assert predictions_ref.shape[0] == X_test.shape[0]
    prediction_dim = predictions_ref.shape[1]

    # Reduce the dimensionality of the predictions if necessary.
    if prediction_dim * 3 > stat_test_dim:
        # Reduce projection_dim such that test_size >= 3 to be able to
        # minimally characterize the variations of the learned prediction
        # function.
        assert stat_test_dim % 3 == 0
        prediction_dim = stat_test_dim // 3
        rp = GaussianRandomProjection(
            n_components=prediction_dim, random_state=random_state
        )
        rp.fit(predictions_ref)
        project = rp.transform
    else:
        project = lambda x: x  # noqa: E731

    test_size = stat_test_dim // prediction_dim

    # XXX: requiring that prediction_dim is a divisor of stat_test_dim. We
    # could alteranatively implement set test_size += 1 and truncates some of
    # the predictions dimensions to match the desired stat_test_dim.
    assert test_size * prediction_dim == stat_test_dim

    # If the following does not hold, we should raise an informative error message to tell
    assert test_size <= X_test.shape[0]

    X_test_diverse_subset = get_diverse_subset(
        X_test, predictions_ref, test_size=test_size
    )
    predictions_weighted_all = []
    predictions_repeated_all = []
    for seed in tqdm(range(n_stochastic_fits)):
        est_weighted = clone(est).set_params(random_state=seed, **extra_params_weighted)
        est_repeated = clone(est).set_params(random_state=seed, **extra_params_repeated)
        est_weighted.fit(X_train, y_train, sample_weight=sample_weight)
        est_repeated.fit(X_resampled_by_weights, y_resampled_by_weights)

        predictions_weighted = compute_predictions(est_weighted, X_test_diverse_subset)
        predictions_repeated = compute_predictions(est_repeated, X_test_diverse_subset)

        predictions_weighted_all.append(project(predictions_weighted))
        predictions_repeated_all.append(project(predictions_repeated))

    predictions_weighted_all = np.stack(predictions_weighted_all)
    predictions_repeated_all = np.stack(predictions_repeated_all)

    return predictions_weighted_all, predictions_repeated_all, X_test_diverse_subset
