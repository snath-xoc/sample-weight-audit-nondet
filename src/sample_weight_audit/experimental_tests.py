from inspect import signature
import numpy as np
import time

from sklearn.base import is_regressor, is_classifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedShuffleSplit,
    permutation_test_score,
)
from sklearn.ensemble import RandomForestClassifier

from .generate_weighted_and_repeated_data import (
    add_faulty_data,
    get_weighted_and_repeated_train_test,
    get_diverse_subset,
)
from .multifit import multifit_over_weighted_and_repeated
from .est_non_deterministic_config import get_config


NON_DEFAULT_PARAMS = get_config()


def test_cv_deterministic(
    est,
    rep_test_size=10,
    n_samples_per_cv_group=1000,
    n_cv_group=3,
    n_features=10,
    **kwargs,
):

    X, y = make_regression(
        n_samples=n_cv_group * n_samples_per_cv_group,
        n_features=n_features,
        noise=100,
        random_state=10,
    )

    (
        X_train,
        y_train,
        X_resampled_by_weights,
        y_resampled_by_weights,
        sample_weight,
        X_test,
        _,
    ) = get_weighted_and_repeated_train_test(X, y, **kwargs)

    est_name = est.__module__ + "." + est.__name__
    extra_params = NON_DEFAULT_PARAMS.get(est_name, {})

    est_init = est(random_state=0, **extra_params)
    est_init.fit(X_train, y_train, sample_weight=sample_weight)
    predictions_init = est_init.predict(X_test)

    X_test_diverse_subset = get_diverse_subset(
        X_test, predictions_init, rep_test_size=rep_test_size
    )

    predictions_weighted = []
    predictions_repeated = []

    for seed in range(100):

        kw_weighted = {"random_state": seed}
        kw_repeated = {"random_state": seed}

        if "cv" in signature(est).parameters:
            groups_weighted = np.r_[
                np.full(n_samples_per_cv_group, 0),
                np.full(n_samples_per_cv_group, 1),
                np.full(n_samples_per_cv_group, 2),
            ]
            splits_weighted = list(LeaveOneGroupOut().split(X, groups=groups_weighted))
            kw_weighted.update({"cv": splits_weighted})

            groups_repeated = np.repeat(
                groups_weighted, sample_weight.astype(int), axis=0
            )
            splits_repeated = list(
                LeaveOneGroupOut().split(X_resampled_by_weights, groups=groups_repeated)
            )
            kw_repeated.update({"cv": splits_repeated})

        est_weighted = est(**kw_weighted)
        est_repeated = est(**kw_repeated)
        est_weighted.fit(X_train, y_train, sample_weight=sample_weight)
        est_repeated.fit(X_resampled_by_weights, y_resampled_by_weights)

        predictions_weighted.append(est_weighted.predict(X_test_diverse_subset))
        predictions_repeated.append(est_repeated.predict(X_test_diverse_subset))

    return np.stack(predictions_weighted), np.stack(predictions_repeated)


def classifier_as_good_as_random(
    est,
    threshold=0.05,
    train_size=500,
    n_samples=1000,
    n_features=10,
    max_repeats=10,
    max_seed=200,
    rep_test_size=10,
    random_state=None,
    **kwargs,
):
    start_time = time.time()

    if is_regressor(est()):
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, random_state=10
        )

        if "RANSAC" in str(est):
            y, inlier_mask = add_faulty_data(X, y)

    elif is_classifier(est()):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            random_state=10,
            n_informative=4,
            n_classes=8,
        )
    print("Fitting models")
    predictions_repeated, predictions_weighted, X_test_diverse_subset = (
        multifit_over_weighted_and_repeated(
            est,
            X,
            y,
            max_seed=max_seed,
            rep_test_size=rep_test_size,
            train_size=train_size,
            max_repeats=max_repeats,
        )
    )

    assert X_test_diverse_subset.shape[0] == predictions_repeated.shape[1]
    assert X_test_diverse_subset.shape[0] == predictions_weighted.shape[1]

    assert predictions_repeated.shape[0] == predictions_weighted.shape[0] == max_seed

    X_test_diverse_subset_tiled = np.tile(X_test_diverse_subset, [max_seed, 1])

    X_clf_repeated = np.hstack(
        (X_test_diverse_subset_tiled, predictions_repeated.reshape(-1, 1))
    )
    X_clf_weighted = np.hstack(
        (X_test_diverse_subset_tiled, predictions_weighted.reshape(-1, 1))
    )
    X_clf = np.vstack((X_clf_repeated, X_clf_weighted))
    y_clf = np.append(
        np.zeros(predictions_repeated.size), np.ones(predictions_weighted.size)
    )

    clf = RandomForestClassifier(**kwargs)

    ## Intialise a simple KFold with 10% of the n_samples (150) splits
    cv = StratifiedShuffleSplit(
        n_splits=30, train_size=300, test_size=100, random_state=random_state
    )

    print("Starting on permutation test")

    ## Perform permutation test using accruacy as a
    ## standard classification problem scoring
    score, perm_scores, pvalue = permutation_test_score(
        clf, X_clf, y_clf, scoring="accuracy", cv=cv
    )

    print(
        "Finished classification test till maximum seed,",
        max_seed,
        "for estimator",
        est,
        "in ----",
        time.time() - start_time,
        "s---",
    )
    print(score)
    print(perm_scores)
    print(pvalue)

    assert pvalue > threshold
