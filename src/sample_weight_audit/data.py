import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.base import is_classifier
from sklearn.utils import check_random_state, shuffle
from sklearn.model_selection import train_test_split

__all__ = [
    "weighted_and_repeated_train_test_split",
    "make_data_for_estimator",
]


def weighted_and_repeated_train_test_split(
    X, y, sample_weight, train_size=500, random_state=None
):
    X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = (
        train_test_split(
            X, y, sample_weight, train_size=train_size, random_state=random_state
        )
    )
    rng = np.random.default_rng(random_state)
    repeated_indices = np.repeat(np.arange(X_train.shape[0]), sample_weight_train)
    shuffled_indices = rng.permutation(len(repeated_indices))
    X_resampled_by_weights = np.take(X_train, repeated_indices, axis=0)[
        shuffled_indices
    ]
    y_resampled_by_weights = np.take(y_train, repeated_indices, axis=0)[
        shuffled_indices
    ]

    return (
        X_train,
        y_train,
        sample_weight_train,
        X_resampled_by_weights,
        y_resampled_by_weights,
        X_test,
        y_test,
        sample_weight_test,
    )


def make_data_for_estimator(
    est, n_samples, n_features, n_classes=3, max_sample_weight=5, random_state=None
):
    # Strategy: sample 2 datasets, each with n_features // 2:
    # - the first one has int(0.8 * n_samples) but mostly zero or one weights.
    # - the second one has the remaining samples but with higher weights.
    #
    # The features of the two datasets are horizontally stacked with random
    # feature values sampled independently from the other dataset. Then the two
    # datasets are vertically stacked and the result is shuffled.
    #
    # The sum of weights of the second dataset is 10 times the sum of weights of
    # the first dataset so that weight aware estimators should mostly ignore the
    # features of the first dataset to learn their prediction function.

    rng = check_random_state(random_state)
    n_samples_sw = int(0.8 * n_samples)  # small weights
    n_samples_lw = n_samples - n_samples_sw  # large weights
    n_features_sw = n_features // 2
    n_features_lw = n_features - n_features_sw

    # Ensure we get enough data points with large weights to be able to
    # get both training and testing data.
    assert n_samples_lw > 30

    # Also ensure that we have at least one feature in each dataset.
    assert n_features_sw >= 1
    assert n_features_lw >= 1

    # Construct the sample weights: mostly zeros and some ones for the first
    # dataset, and some random integers larger than one for the second dataset.
    sample_weight_sw = np.where(rng.random(n_samples_sw) < 0.2, 1, 0)
    sample_weight_lw = rng.randint(2, max_sample_weight, size=n_samples_lw)
    total_weight_sum = np.sum(sample_weight_sw) + np.sum(sample_weight_lw)
    assert np.sum(sample_weight_sw) < 0.3 * total_weight_sum

    if not is_classifier(est):
        X_sw, y_sw = make_regression(
            n_samples=n_samples_sw,
            n_features=n_features_sw,
            random_state=rng,
        )
        X_lw, y_lw = make_regression(
            n_samples=n_samples_lw,
            n_features=n_features_lw,
            random_state=rng,  # rng is different because mutated
        )
    else:
        X_sw, y_sw = make_classification(
            n_samples=n_samples_sw,
            n_features=n_features_sw,
            n_informative=n_features_sw,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            random_state=rng,
        )
        X_lw, y_lw = make_classification(
            n_samples=n_samples_lw,
            n_features=n_features_lw,
            n_informative=n_features_lw,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            random_state=rng,  # rng is different because mutated
        )

    # Horizontally pad the features with features values marginally sampled
    # from the other dataset.
    pad_sw_idx = rng.choice(n_samples_lw, size=n_samples_sw, replace=True)
    X_sw_padded = np.hstack([X_sw, np.take(X_lw, pad_sw_idx, axis=0)])

    pad_lw_idx = rng.choice(n_samples_sw, size=n_samples_lw, replace=True)
    X_lw_padded = np.hstack([np.take(X_sw, pad_lw_idx, axis=0), X_lw])

    # Vertically stack the two datasets and shuffle them.
    X = np.concatenate([X_sw_padded, X_lw_padded], axis=0)
    y = np.concatenate([y_sw, y_lw])
    sample_weight = np.concatenate([sample_weight_sw, sample_weight_lw])
    return shuffle(X, y, sample_weight, random_state=rng)
