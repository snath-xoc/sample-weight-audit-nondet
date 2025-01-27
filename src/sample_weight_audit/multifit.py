from inspect import signature
import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

from scipy.sparse import csr_matrix, csr_array
from sklearn.base import clone, is_regressor, is_classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import LeaveOneGroupOut

from .generate_weighted_and_repeated_data import (
    get_weighted_and_repeated_train_test,
    get_diverse_subset,
)
from .est_non_deterministic_config import get_config


NON_DEFAULT_PARAMS = get_config()


def get_extra_params(est):
    est_class = est.__class__
    est_name = est_class.__module__ + "." + est_class.__name__
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
    if is_regressor(est):
        return est_init.predict(X_test)
    elif is_classifier(est):
        if hasattr(est_init, "predict_proba"):
            # TODO: when predicting probabilities, the last column is redundant
            # because of sum-to-one constraint and could therefore be
            # discarded. But this is not the case when using decision_function.
            # We should probably trim it here instead of doing this later.
            return est_init.predict_proba(X_test)
        else:
            return est_init.decision_function(X_test)

    elif hasattr(est_init, "transform"):
        return est_init.transform(X_test)

    else:
        raise NotImplementedError(f"Estimator type not supported: {est}")


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
    # TODO: remove the SVM specific wrapper and instead, let's report SVC
    # models as failing but also add the CalibratedClassifierCV equivalent to
    # the list of scikit-learn models to test in the notebooks.
    if "probability" in signature(est.__init__).parameters and calibrated:
        cv = None
        cv_rep = None
        if "cv" in extra_params:
            cv = extra_params.pop("cv")
            cv_rep = extra_params_rep.pop("cv")
        est_weighted = CalibratedClassifierCV(
            estimator=clone(est).set_params(random_state=seed, **extra_params),
            cv=cv,
            ensemble=False,
        )
        est_repeated = CalibratedClassifierCV(
            estimator=clone(est).set_params(random_state=seed, **extra_params_rep),
            cv=cv_rep,
            ensemble=False,
        )
    else:
        est_weighted = clone(est).set_params(random_state=seed, **extra_params)
        est_repeated = clone(est).set_params(random_state=seed, **extra_params_rep)

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
    ## TODO: currently output is flattened but should be changed to
    ## handle all the output dimensions individually

    # TODO: remove estimator specific branches and always rely on the `transform`
    # method.
    if isinstance(est, KBinsDiscretizer):
        predictions_weighted = np.stack(est_weighted.bin_edges_)
        predictions_repeated = np.stack(est_repeated.bin_edges_)
    elif isinstance(est, RandomTreesEmbedding):
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
    if is_regressor(est):
        preds_weighted, preds_repeated = get_regression_results(
            est_weighted, est_repeated, X_test_diverse_subset
        )
        # Reshape to 2D to match classifier and tranformer output shapes.
        return preds_weighted.reshape(-1, 1), preds_repeated.reshape(-1, 1)
    elif is_classifier(est):
        return get_classifier_results(
            est_weighted, est_repeated, X_test_diverse_subset, n_classes
        )
    elif hasattr(est, "transform"):
        return get_transformer_results(
            est, est_weighted, est_repeated, X_test_diverse_subset
        )
    else:
        raise NotImplementedError(f"Estimator type not supported: {est}")


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

    if "cv" in signature(est.__init__).parameters:
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

    est_init = clone(est).set_params(random_state=0, **extra_params)

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
