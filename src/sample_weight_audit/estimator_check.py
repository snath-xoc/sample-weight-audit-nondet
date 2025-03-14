from dataclasses import dataclass

import numpy as np
from inspect import signature
from sklearn.base import clone, is_classifier, is_regressor, is_clusterer
from sklearn.utils.multiclass import type_of_target
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    log_loss,
    roc_auc_score,
    adjusted_rand_score,
    average_precision_score,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm


from .data import (
    make_data_for_estimator,
    weighted_and_repeated_train_test_split,
)
from .statistical_testing import run_1d_test


@dataclass
class EquivalenceTestResult:
    estimator_name: str
    test_name: str
    p_value: float
    scores_weighted: np.ndarray
    scores_repeated: np.ndarray
    deterministic_flag: bool

    def __repr__(self):
        return (
            "EquivalenceTestResult("
            f"estimator_name={self.estimator_name!r}, "
            f"test_name={self.test_name!r}, "
            f"p_value={self.p_value}, "
            f"deterministic_flag={self.deterministic_flag}, "
        )

    def to_dict(self):
        return {
            "estimator_name": self.estimator_name,
            "test_name": self.test_name,
            "p_value": self.p_value,
            "scores_weighted": self.scores_weighted,
            "scores_repeated": self.scores_repeated,
            "deterministic_flag": self.deterministic_flag,
        }


def check_weighted_repeated_estimator_fit_equivalence(
    est,
    test_name="kstest",
    n_samples_per_cv_group=100,
    n_cv_group=3,
    n_features=10,
    n_classes=3,
    max_sample_weight=10,
    n_stochastic_fits=300,
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

    scores_weighted, scores_repeated, predictions_weighted, predictions_repeated = (
        multifit_over_weighted_and_repeated(
            est,
            n_features=n_features,
            n_classes=n_classes,
            n_stochastic_fits=n_stochastic_fits,
            n_samples_per_cv_group=n_samples_per_cv_group,
            n_cv_group=n_cv_group,
            max_sample_weight=max_sample_weight,
            random_state=random_state,
        )
    )

    assert scores_weighted.ndim == 1  # (n_stochastic_fits,)
    assert scores_weighted.shape == scores_repeated.shape

    deterministic_flag = False
    if predictions_weighted.dtype != np.int32:
        diffs = predictions_weighted.max(axis=0) - predictions_weighted.min(axis=0)
        if np.all(diffs < np.finfo(diffs.dtype).eps):
            deterministic_flag = True

    if predictions_repeated.dtype != np.int32:
        diffs = predictions_repeated.max(axis=0) - predictions_repeated.min(axis=0)
        if np.all(diffs < np.finfo(diffs.dtype).eps):
            deterministic_flag = True

    assert scores_weighted.ndim == 1
    assert scores_weighted.shape[0] == n_stochastic_fits
    # assert math.prod(predictions_weighted.shape[1:]) == stat_test_dim
    assert scores_repeated.shape == scores_weighted.shape

    data_to_test_weighted = scores_weighted
    data_to_test_repeated = scores_repeated
    # Iterate of all statistical test dimensions and compute p-values
    # for each dimension.
    pvalue = run_1d_test(data_to_test_weighted, data_to_test_repeated, test_name).pvalue

    return EquivalenceTestResult(
        est.__class__.__name__,
        test_name,
        pvalue,
        scores_weighted,
        scores_repeated,
        deterministic_flag,
    )


def get_cv_params(
    n_cv_group,
    n_samples_per_cv_group,
    X_train,
    X_resampled_by_weights,
    sample_weight,
):
    groups_weighted = np.hstack(
        ([np.full(n_samples_per_cv_group, i) for i in range(n_cv_group)])
    )
    splits_weighted = list(LeaveOneGroupOut().split(X_train, groups=groups_weighted))
    extra_params_weighted = {"cv": splits_weighted}

    groups_repeated = np.repeat(groups_weighted, sample_weight.astype(int), axis=0)
    splits_repeated = list(
        LeaveOneGroupOut().split(X_resampled_by_weights, groups=groups_repeated)
    )
    extra_params_repeated = {"cv": splits_repeated}
    return extra_params_weighted, extra_params_repeated


def score_estimator(est, X, y):

    if is_regressor(est):
        preds = est.predict(X)
        return mean_squared_error(preds, y), preds
    elif is_classifier(est):
        if hasattr(est, "predict_proba"):
            preds = est.predict_proba(X)
            return log_loss(y, preds), preds
        else:
            preds = est.decision_function(X)
            y_type = type_of_target(y)
            if y_type not in ("binary", "multilabel-indicator"):
                return average_precision_score(y, preds), preds
            else:
                return roc_auc_score(preds, y), preds
    elif is_clusterer(est):
        preds = est.predict(X)
        return adjusted_rand_score(preds, y), preds
    else:
        raise NotImplementedError(f"Estimator type not supported: {est}")


def check_pipeline_and_fit(est, X, y, sample_weight=None, seed=None):
    if (
        not is_classifier(est)
        and not is_regressor(est)
        and not is_clusterer(est)
        and hasattr(est, "transform")
    ):
        est = Pipeline(
            [
                ("transformer", est),
                ("ridge", Ridge(random_state=seed, fit_intercept=False)),
            ]
        )
        est = est.fit(
            X,
            y,
            transformer__sample_weight=sample_weight,
            ridge__sample_weight=sample_weight,
        )
    else:
        est = est.fit(X, y, sample_weight=sample_weight)
    return est


def multifit_over_weighted_and_repeated(
    est,
    n_stochastic_fits=200,
    n_samples_per_cv_group=100,
    n_cv_group=3,
    n_features=10,
    n_classes=3,
    test_pool_size=1000,
    max_sample_weight=5,
    random_state=None,
):
    if "cv" in est.get_params():
        use_cv = True
        effective_train_size = n_samples_per_cv_group * n_cv_group
    else:
        use_cv = False
        effective_train_size = n_samples_per_cv_group

    X, y, sample_weight = make_data_for_estimator(
        est,
        effective_train_size + test_pool_size,
        n_features=n_features,
        n_classes=n_classes,
        max_sample_weight=max_sample_weight,
        random_state=random_state,
    )

    (
        X_train,
        y_train,
        sample_weight_train,
        X_resampled_by_weights,
        y_resampled_by_weights,
        X_test,
        y_test,
        _,
    ) = weighted_and_repeated_train_test_split(
        X,
        y,
        sample_weight=sample_weight,
        train_size=effective_train_size,
        random_state=random_state,
    )

    if use_cv:
        extra_params_weighted, extra_params_repeated = get_cv_params(
            n_cv_group,
            n_samples_per_cv_group,
            X_train,
            X_resampled_by_weights,
            sample_weight_train,
        )
    else:
        extra_params_weighted = {}
        extra_params_repeated = {}

    # Perform one reference fit to inspect the predictions dimensions.
    est_ref = clone(est).set_params(**extra_params_weighted)
    if "random_state" in signature(est.fit).parameters:
        est_ref.set_params(random_state=0)
    est_ref = check_pipeline_and_fit(est_ref, X_train, y_train, sample_weight_train)

    scores_weighted_all = []
    scores_repeated_all = []
    predictions_weighted_all = []
    predictions_repeated_all = []
    for seed in tqdm(range(n_stochastic_fits)):
        if "random_state" in signature(est.__init__).parameters:
            est_weighted = clone(est).set_params(
                random_state=seed, **extra_params_weighted
            )
            # Use different random seeds for the other group of fits to avoid
            # introducing an unwanted statistical dependency.
            est_repeated = clone(est).set_params(
                random_state=seed + n_stochastic_fits, **extra_params_repeated
            )
        else:
            est_weighted = clone(est).set_params(**extra_params_weighted)
            est_repeated = clone(est).set_params(**extra_params_repeated)

        est_weighted = check_pipeline_and_fit(
            est_weighted,
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            seed=seed,
        )
        est_repeated = check_pipeline_and_fit(
            est_repeated,
            X_resampled_by_weights,
            y_resampled_by_weights,
            seed=seed,
        )

        scores_weighted, preds_weighted = score_estimator(est_weighted, X_test, y_test)
        scores_repeated, preds_repeated = score_estimator(est_repeated, X_test, y_test)

        scores_weighted_all.append(scores_weighted)
        scores_repeated_all.append(scores_repeated)
        predictions_weighted_all.append(preds_weighted)
        predictions_repeated_all.append(preds_repeated)

    scores_weighted_all = np.stack(scores_weighted_all)
    scores_repeated_all = np.stack(scores_repeated_all)
    predictions_weighted_all = np.stack(predictions_weighted_all).reshape(
        n_stochastic_fits, -1
    )
    predictions_repeated_all = np.stack(predictions_repeated_all).reshape(
        n_stochastic_fits, -1
    )

    return (
        scores_weighted_all,
        scores_repeated_all,
        predictions_weighted_all,
        predictions_repeated_all,
    )
