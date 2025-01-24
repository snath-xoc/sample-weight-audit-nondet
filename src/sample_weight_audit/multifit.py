from inspect import signature
import numpy as np
from tqdm import tqdm

from scipy.sparse import csr_matrix, csr_array
from sklearn.base import is_regressor, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneGroupOut

from .generate_weighted_and_repeated_data import (
    get_weighted_and_repeated_train_test,
    get_diverse_subset,
)
from .est_non_deterministic_config import get_config


NON_DEFAULT_PARAMS = get_config()


def get_extra_params(est):
    est_name = est.__module__ + "." + est.__name__
    extra_params = NON_DEFAULT_PARAMS.get(est_name, {})
    extra_params_rep = extra_params.copy()

    return extra_params, extra_params_rep


def get_cv_split(
    est,
    extra_params,
    extra_params_rep,
    train_size,
    n_cv_group,
    X_train,
    X_resampled_by_weights,
    sample_weight,
):

    groups_weighted = np.hstack(([np.full(train_size, i) for i in range(n_cv_group)]))
    splits_weighted = list(LeaveOneGroupOut().split(X_train, groups=groups_weighted))

    groups_repeated = np.repeat(groups_weighted, sample_weight.astype(int), axis=0)
    splits_repeated = list(
        LeaveOneGroupOut().split(X_resampled_by_weights, groups=groups_repeated)
    )

    extra_params["cv"] = splits_weighted
    extra_params_rep["cv"] = splits_repeated

    return extra_params, extra_params_rep


def get_initial_predictions(est, est_init, X_test, n_classes=None):

    if is_regressor(est()):
        predictions_init = est_init.predict(X_test)
    elif is_classifier(est()):
        ## TO DO: classes from make_classifications are not ordinal so should return
        # £ predicted probabilitied
        if hasattr(est_init, "predict_proba"):
            predictions_init = est_init.predict_proba(X_test)
        else:
            predictions_init = est_init.decision_function(X_test)

    else:
        predictions_init = X_test

    return predictions_init


def get_est_weighted_and_repeated(
    est,
    seed,
    extra_params,
    extra_params_rep,
    X_train,
    y_train,
    X_resampled_by_weights,
    y_resampled_by_weights,
    sample_weight,
    calibrated=False,
):
    if "probability" in signature(est).parameters and calibrated:
        cv = None
        cv_rep = None
        if "cv" in extra_params:
            cv = extra_params.pop("cv")
            cv_rep = extra_params_rep.pop("cv")
        est_weighted = CalibratedClassifierCV(
            estimator=est(random_state=seed, **extra_params), cv=cv, ensemble=False
        )
        est_repeated = CalibratedClassifierCV(
            estimator=est(random_state=seed, **extra_params_rep),
            cv=cv_rep,
            ensemble=False,
        )
    else:
        est_weighted = est(random_state=seed, **extra_params)
        est_repeated = est(random_state=seed, **extra_params_rep)

    ## for case of transformers only X is fit on
    if not is_regressor(est()) and not is_classifier(est()):
        est_weighted.fit(X_train, sample_weight=sample_weight)
        est_repeated.fit(X_resampled_by_weights)
    else:
        est_weighted.fit(X_train, y_train, sample_weight=sample_weight)
        est_repeated.fit(X_resampled_by_weights, y_resampled_by_weights)

    return est_weighted, est_repeated


def get_regression_results(est_weighted, est_repeated, X_test_diverse_subset):

    return est_weighted.predict(X_test_diverse_subset), est_repeated.predict(
        X_test_diverse_subset
    )


def get_classifier_results(
    est_weighted, est_repeated, X_test_diverse_subset, n_classes
):
    ## TO DO: n_classes not ordinal from make_classification
    ## Need to change to handle all predicted probability
    ## across n_classes
    if hasattr(est_weighted, "predict_proba"):
        ## We throw away first output dimension of predict proba since it doesn't
        ## mean anything
        predictions_weighted = est_weighted.predict_proba(X_test_diverse_subset)[:, :-1]

        predictions_repeated = est_repeated.predict_proba(X_test_diverse_subset)[:, :-1]

    else:
        predictions_weighted = est_weighted.decision_function(X_test_diverse_subset)

        predictions_repeated = est_repeated.decision_function(X_test_diverse_subset)

    return predictions_weighted, predictions_repeated


def get_transformer_results(est, est_weighted, est_repeated, X_test_diverse_subset):

    ## TO DO: currently output is flattened but should be changed to
    ## handle all the output dimensions individually
    if est.__name__ == "KBinsDiscretizer":
        predictions_weighted = np.stack(est_weighted.bin_edges_)
        predictions_repeated = np.stack(est_repeated.bin_edges_)
    elif est.__name__ == "RandomTreesEmbedding":
        predictions_weighted = est_weighted.transform(X_test_diverse_subset)

        predictions_repeated = est_repeated.transform(X_test_diverse_subset)

    else:
        predictions_weighted = np.asarray(est_weighted.transform(X_test_diverse_subset))

        predictions_repeated = np.asarray(est_repeated.transform(X_test_diverse_subset))

    if isinstance(predictions_weighted, csr_matrix) or isinstance(
        predictions_weighted, csr_array
    ):
        predictions_weighted = predictions_weighted.toarray()
    if isinstance(predictions_repeated, csr_matrix) or isinstance(
        predictions_repeated, csr_array
    ):
        predictions_repeated = predictions_repeated.toarray()

    return predictions_weighted, predictions_repeated


def get_predictions(
    est, est_weighted, est_repeated, X_test_diverse_subset, n_classes=None
):
    if is_regressor(est()):
        predictions_weighted, predictions_repeated = get_regression_results(
            est_weighted, est_repeated, X_test_diverse_subset
        )

    elif is_classifier(est()):

        predictions_weighted, predictions_repeated = get_classifier_results(
            est_weighted, est_repeated, X_test_diverse_subset, n_classes
        )

    else:
        predictions_weighted, predictions_repeated = get_transformer_results(
            est, est_weighted, est_repeated, X_test_diverse_subset
        )
    return predictions_weighted, predictions_repeated


def multifit_over_weighted_and_repeated(
    est,
    X,
    y,
    max_seed=200,
    rep_test_size=10,
    n_classes=None,
    train_size=100,
    n_cv_group=3,
    calibrated=False,
    **kwargs,
):
    extra_params, extra_params_rep = get_extra_params(est)
    if "ignore_sample_weight" in kwargs.keys():
        print(kwargs["ignore_sample_weight"])
        extra_params["ignore_sample_weight"] = kwargs.pop("ignore_sample_weight")
        extra_params_rep["ignore_sample_weight"] = extra_params["ignore_sample_weight"]

    (
        X_train,
        y_train,
        X_resampled_by_weights,
        y_resampled_by_weights,
        sample_weight,
        X_test,
        _,
    ) = get_weighted_and_repeated_train_test(
        X, y, train_size=train_size * n_cv_group, **kwargs
    )

    if "cv" in signature(est).parameters:
        extra_params, extra_params_rep = get_cv_split(
            est,
            extra_params,
            extra_params_rep,
            train_size,
            n_cv_group,
            X_train,
            X_resampled_by_weights,
            sample_weight,
        )

    est_init = est(random_state=0, **extra_params)

    est_init.fit(X_train, y_train, sample_weight=sample_weight)

    predictions_init = get_initial_predictions(
        est, est_init, X_test, n_classes=n_classes
    )

    X_test_diverse_subset = get_diverse_subset(
        X_test, predictions_init, rep_test_size=rep_test_size
    )
    predictions_weighted_all = []
    predictions_repeated_all = []
    ##add progress bar
    for seed in tqdm(range(max_seed)):

        est_weighted, est_repeated = get_est_weighted_and_repeated(
            est,
            seed,
            extra_params,
            extra_params_rep,
            X_train,
            y_train,
            X_resampled_by_weights,
            y_resampled_by_weights,
            sample_weight,
            calibrated=calibrated,
        )

        predictions_weighted, predictions_repeated = get_predictions(
            est, est_weighted, est_repeated, X_test_diverse_subset, n_classes=n_classes
        )
        predictions_weighted_all.append(predictions_weighted)
        predictions_repeated_all.append(predictions_repeated)

    predictions_weighted_all = np.stack(predictions_weighted_all)
    predictions_repeated_all = np.stack(predictions_repeated_all)

    return predictions_weighted_all, predictions_repeated_all, X_test_diverse_subset
