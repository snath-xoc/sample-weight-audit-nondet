import numpy as np
from scipy.sparse import csr_matrix, csr_array
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.base import is_regressor, is_classifier
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(1000)

__all__ = [
    "add_faulty_data",
    "get_diverse_subset",
    "get_weighted_and_repeated_train_test",
    "get_estimator_dataset",
]


def add_faulty_data(X, y):
    # Add some faulty data
    outliers = rng.choice(X.shape[0], 30, replace=False)
    y[outliers] += 1000 + rng.rand(len(outliers)) * 10
    inlier_mask = np.ones(X.shape[0], dtype=bool)
    inlier_mask[outliers] = False

    return y, inlier_mask


def get_diverse_subset(X_test, predictions, rep_test_size=10):
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


def get_estimator_dataset(
    est, n_samples_per_cv_group, n_cv_group, n_features, n_classes=None
):
    if is_regressor(est):
        X, y = make_regression(
            n_samples=n_samples_per_cv_group * n_cv_group,
            n_features=n_features,
            noise=100,
            random_state=10,
        )
        y, inlier_mask = add_faulty_data(X, y)

    elif is_classifier(est):
        X, y = make_classification(
            n_samples=n_samples_per_cv_group * n_cv_group,
            n_features=n_features,
            random_state=10,
            n_informative=4,
            n_classes=n_classes,
        )

    else:
        centres = np.array([[0, 0, 0], [0, 5, 5], [3, 1, 1], [2, 4, 4], [100, 8, 800]])
        X, _ = make_blobs(
            n_samples=n_samples_per_cv_group * n_cv_group,
            cluster_std=0.5,
            centers=centres,
            random_state=10,
        )
        y = None
    return X, y
