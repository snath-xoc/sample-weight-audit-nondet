import time
from tqdm import tqdm

from inspect import signature

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest, mannwhitneyu, ttest_ind
from scipy.stats.mstats import compare_medians_ms
from scipy.sparse import csr_matrix, csr_array

from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import is_regressor, is_classifier
from sklearn.model_selection import (
    permutation_test_score,
    train_test_split,
    StratifiedShuffleSplit,
    LeaveOneGroupOut,
)
from sklearn.ensemble import RandomForestClassifier

rng = np.random.RandomState(1000)
NON_DEFAULT_PARAMS = {
    "sklearn.linear_model._coordinate_descent.ElasticNet": {"selection": "random"},
    "sklearn.tree._classes.DecisionTreeRegressor": {"max_features": 0.5},
    "sklearn.linear_model._coordinate_descent.ElasticNetCV": {"selection": "cyclic"},
    "sklearn.ensemble._gb.GradientBoostingRegressor": {"max_features": 0.5},
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor": {
        "max_features": 0.5
    },
    "sklearn.linear_model._coordinate_descent.Lasso": {"selection": "random"},
    "sklearn.linear_model._coordinate_descent.LassoCV": {"selection": "cyclic"},
    "sklearn.svm._classes.LinearSVR": {"dual": "auto"},
    "sklearn.linear_model._ridge.Ridge": {"solver": "sag"},
    "sklearn.ensemble._forest.RandomForestRegressor": {"max_features": 0.5},
    "sklearn.tree._classes.DecisionTreeClassifier": {"max_features": 0.5},
    "sklearn.tree._classes.DecisionTreeClassifier": {"max_features": 0.5},
    "sklearn.dummy.DummyClassifier": {"strategy": "stratified"},
    "sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier": {
        "max_features": 0.5
    },
    "sklearn.ensemble._gb.GradientBoostingClassifier": {"max_features": 0.5},
    "sklearn.svm._classes.LinearSVC": {"dual": True},
    "sklearn.linear_model._logistic.LogisticRegression": {
        "dual": True,
        "solver": "liblinear",
        "max_iter": 10000,
    },
    "sklearn.linear_model._logistic.LogisticRegressionCV": {
        "dual": True,
        "solver": "liblinear",
        "max_iter": 10000,
    },
    "sklearn.svm._classes.NuSVC": {"probability": False},
    "sklearn.linear_model._ridge.RidgeClassifier": {"solver": "saga"},
    "sklearn.svm._classes.SVC": {"probability": False},
    "sklearn.preprocessing._discretization.KBinsDiscretizer": {
        "subsample": 50,
        "encode": "ordinal",
        "strategy": "kmeans",
    },
    "sklearn.cluster._kmeans.MiniBatchKMeans": {"reassignment_ratio": 0.9},
    "sklearn.ensemble._forest.RandomTreesEmbedding": {
        "sparse_output": False,
        "n_estimators": 20,
    },
}


def add_faulty_data(X, y):
    # Add some faulty data
    outliers = rng.choice(X.shape[0], 30, replace=False)
    y[outliers] += 1000 + rng.rand(len(outliers)) * 10
    inlier_mask = np.ones(X.shape[0], dtype=bool)
    inlier_mask[outliers] = False

    return y, inlier_mask


def get_representative_sample(X_test, predictions, rep_test_size=10):
    if isinstance(predictions, csr_matrix) or isinstance(predictions, csr_array):
        predictions = predictions.toarray()
    if predictions.ndim == 2:
        predictions = predictions.max(axis=1)
    prediction_ranks = predictions.argsort()
    test_indices = prediction_ranks[
        np.linspace(0, len(prediction_ranks) - 1, rep_test_size).astype(np.int32)
    ]

    return X_test[test_indices]


def get_weighted_and_repeated_train_test(X, y, train_size=500, max_repeats=10):

    issparse = False
    if isinstance(X, csr_array):
        X = X.toarray()
        issparse = True

    if y is None:
        y = X

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    sample_weight = rng.randint(0, max_repeats + 1, size=X_train.shape[0])
    X_resampled_by_weights = np.repeat(X_train, sample_weight, axis=0)
    y_resampled_by_weights = np.repeat(y_train, sample_weight, axis=0)

    if y is None:
        y_train = None
        y_resampled_by_weights = None
        y_test = None

    if issparse:
        X_train = csr_array(X_train)
        X_resampled_by_weights = csr_array(X_resampled_by_weights)
        X_test = csr_array(X_test)

    return (
        X_train,
        y_train,
        X_resampled_by_weights,
        y_resampled_by_weights,
        sample_weight,
        X_test,
        y_test,
    )


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

    est_name = est.__module__ + "." + est.__name__
    extra_params = NON_DEFAULT_PARAMS.get(est_name, {})
    extra_params_rep = extra_params.copy()

    if "cv" in signature(est).parameters:

        groups_weighted = np.hstack(
            ([np.full(train_size, i) for i in range(n_cv_group)])
        )
        splits_weighted = list(
            LeaveOneGroupOut().split(X_train, groups=groups_weighted)
        )

        groups_repeated = np.repeat(groups_weighted, sample_weight.astype(int), axis=0)
        splits_repeated = list(
            LeaveOneGroupOut().split(X_resampled_by_weights, groups=groups_repeated)
        )

        extra_params["cv"] = splits_weighted
        extra_params_rep["cv"] = splits_repeated

    ## create mapping between classes and non-default hyperparameters

    est_init = est(random_state=0, **extra_params)
    est_init.fit(X_train, y_train, sample_weight=sample_weight)

    if is_regressor(est):
        predictions_init = est_init.predict(X_test)
    elif is_classifier(est):

        try:
            predictions_init = est_init.predict_proba(X_test) @ np.arange(n_classes)
        except:
            predictions_init = est_init.decision_function(X_test) @ np.arange(n_classes)

    else:
        predictions_init = X_test

    X_test_representative = get_representative_sample(
        X_test, predictions_init, rep_test_size=rep_test_size
    )
    predictions_weighted = []
    predictions_repeated = []
    ##add progress bar
    for seed in tqdm(range(max_seed)):

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

        if is_regressor(est):
            est_weighted.fit(X_train, y_train, sample_weight=sample_weight)
            est_repeated.fit(X_resampled_by_weights, y_resampled_by_weights)

            predictions_weighted.append(est_weighted.predict(X_test_representative))
            predictions_repeated.append(est_repeated.predict(X_test_representative))

        elif is_classifier(est):

            est_weighted.fit(X_train, y_train, sample_weight=sample_weight)
            est_repeated.fit(X_resampled_by_weights, y_resampled_by_weights)

            try:
                predictions_weighted.append(
                    est_weighted.predict_proba(X_test_representative)
                    @ np.arange(n_classes)
                )
                predictions_repeated.append(
                    est_repeated.predict_proba(X_test_representative)
                    @ np.arange(n_classes)
                )
            except:
                predictions_weighted.append(
                    est_weighted.decision_function(X_test_representative)
                    @ np.arange(n_classes)
                )
                predictions_repeated.append(
                    est_repeated.decision_function(X_test_representative)
                    @ np.arange(n_classes)
                )
        else:
            est_weighted.fit(X_train, sample_weight=sample_weight)
            est_repeated.fit(X_resampled_by_weights)

            if est.__name__ == "KBinsDiscretizer":
                predictions_weighted.append(np.stack(est_weighted.bin_edges_).flatten())
                predictions_repeated.append(np.stack(est_repeated.bin_edges_).flatten())
            elif est.__name__ == "RandomTreesEmbedding":
                predictions_weighted.append(
                    np.stack(est_weighted.apply(X_test_representative).mean(axis=1))
                )
                predictions_repeated.append(
                    np.stack(est_repeated.apply(X_test_representative).mean(axis=1))
                )
            else:
                predictions_weighted.append(
                    np.asarray(est_weighted.transform(X_test_representative)).flatten()
                )
                predictions_repeated.append(
                    np.asarray(est_repeated.transform(X_test_representative)).flatten()
                )

    predictions_repeated = np.stack(predictions_repeated)
    predictions_weighted = np.stack(predictions_weighted)

    return predictions_repeated, predictions_weighted, X_test_representative


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

    X_test_representative = get_representative_sample(
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

        predictions_weighted.append(est_weighted.predict(X_test_representative))
        predictions_repeated.append(est_repeated.predict(X_test_representative))

    return np.stack(predictions_weighted), np.stack(predictions_repeated)


def paired_test(
    est,
    test="kstest",
    threshold=0.05,
    correct_threshold=False,
    random_rejection_level=0.1,
    train_size=100,
    n_samples_per_cv_group=200,
    n_cv_group=3,
    n_features=10,
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
    n_classes = None

    if is_regressor(est):
        X, y = make_regression(
            n_samples=n_samples_per_cv_group * n_cv_group,
            n_features=n_features,
            noise=100,
            random_state=10,
        )
        y, inlier_mask = add_faulty_data(X, y)

    elif is_classifier(est):
        n_classes = 8
        X, y = make_classification(
            n_samples=n_samples_per_cv_group * n_cv_group,
            n_features=n_features,
            random_state=10,
            n_informative=4,
            n_classes=n_classes,
        )

    else:
        centres = np.array([[0, 0], [0, 5], [3, 1], [2, 4], [100, 8]])
        X, _ = make_blobs(
            n_samples=n_samples_per_cv_group * n_cv_group,
            cluster_std=0.5,
            centers=centres,
            random_state=10,
        )
        y = None

    if issparse:
        X = csr_array(X)

    predictions_repeated, predictions_weighted, _ = multifit_over_weighted_and_repeated(
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
    print(predictions_repeated.shape)
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
    median_difference_ms = []
    median_difference = []

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

        median_difference_ms.append(compare_medians_ms(pred, pred_ref))
        median_difference.append(np.median(pred) - np.median(pred_ref))

    print(
        "Finished looping till the maximum random state,",
        max_seed,
        "for estimator",
        est,
        "in ----",
        time.time() - start_time,
        "s---",
    )
    print("Average difference in medians is:", np.mean(median_difference))
    print("Minimum p-values: ", np.array(p_vals).min())

    return {
        "Name": est.__name__,
        "p_values": p_vals,
        "min_p_value": np.array(p_vals).min(),
        "avg_p_value": np.nanmean(p_vals),
    }


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

    if is_regressor(est):
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, random_state=10
        )

        if "RANSAC" in str(est):
            y, inlier_mask = add_faulty_data(X, y)

    elif is_classifier(est):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            random_state=10,
            n_informative=4,
            n_classes=8,
        )
    print("Fitting models")
    predictions_repeated, predictions_weighted, X_test_representative = (
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

    assert X_test_representative.shape[0] == predictions_repeated.shape[1]
    assert X_test_representative.shape[0] == predictions_weighted.shape[1]

    assert predictions_repeated.shape[0] == predictions_weighted.shape[0] == max_seed

    X_test_representative_tiled = np.tile(X_test_representative, [max_seed, 1])

    X_clf_repeated = np.hstack(
        (X_test_representative_tiled, predictions_repeated.reshape(-1, 1))
    )
    X_clf_weighted = np.hstack(
        (X_test_representative_tiled, predictions_weighted.reshape(-1, 1))
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

    # if "Classifier" in str(est):
    #   assert_array_equal(predictions_weighted, predictions_repeated)
