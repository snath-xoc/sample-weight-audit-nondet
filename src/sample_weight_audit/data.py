import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.base import is_classifier, is_clusterer
from sklearn.utils import check_random_state, shuffle
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_y,
    _enforce_estimator_tags_X,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

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
    repeated_indices = np.repeat(np.arange(X_train.shape[0]), sample_weight_train)
    X_resampled_by_weights = np.take(X_train, repeated_indices, axis=0)
    y_resampled_by_weights = np.take(y_train, repeated_indices, axis=0)
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
    est,
    n_samples,
    n_features,
    n_classes=3,
    max_sample_weight=5,
    n_samp_eq_sw_sum=False,
    random_state=None,
):
    # Strategy: sample 2 datasets, each with n_features // 2:
    # - the first one has int(0.9 * n_samples) but mostly zero or one weights.
    # - the second one has the remaining samples but with higher weights.
    # - the sum of all weights equals n_samples so that estimators that have
    #   hyperparameters that depend on the sum of weights should behave the
    #   same when fitted with sample weights or repeated samples.
    #
    # The features of the two datasets are horizontally stacked with random
    # feature values sampled independently from the other dataset. Then the two
    # datasets are vertically stacked and the result is shuffled.
    #
    # The sum of weights of the second dataset is 10 times the sum of weights of
    # the first dataset so that weight aware estimators should mostly ignore the
    # features of the first dataset to learn their prediction function.

    rng = check_random_state(random_state)
    if n_samp_eq_sw_sum:
        n_samples_sw = int(0.9 * n_samples)  # small weights
    else:
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
    # Let's start with the second dataset, which has larger weights:
    if n_samp_eq_sw_sum:
        sample_weight_lw = rng.randint(low=3, high=11, size=n_samples_lw)
        sample_weigh_lw_sum = np.sum(sample_weight_lw)
        assert sample_weigh_lw_sum < n_samples
        assert n_samples_sw > n_samples - sample_weigh_lw_sum

        # Allocate 0 or 1 weights for the first dataset, such that the sum of all
        # weights matches the total number of samples.
        sample_weight_sw = np.zeros(n_samples_sw, dtype=sample_weight_lw.dtype)
        weight_one_indices = rng.choice(
            n_samples_sw, size=n_samples - sample_weigh_lw_sum, replace=False
        )
        sample_weight_sw[weight_one_indices] = 1
        assert np.sum(sample_weight_sw) + sample_weigh_lw_sum == n_samples

        sample_weight_sw_sum = np.sum(sample_weight_sw)
        total_weight_sum = sample_weight_sw_sum + sample_weigh_lw_sum
        assert sample_weight_sw_sum < 0.4 * total_weight_sum, (
            sample_weight_sw_sum,
            total_weight_sum,
        )
    else:
        sample_weight_sw = np.where(rng.random(n_samples_sw) < 0.2, 1, 0)
        sample_weight_lw = rng.randint(2, max_sample_weight, size=n_samples_lw)
        total_weight_sum = np.sum(sample_weight_sw) + np.sum(sample_weight_lw)
        assert np.sum(sample_weight_sw) < 0.4 * total_weight_sum

    if is_classifier(est) or is_clusterer(est):
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
    else:
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

    # Horizontally pad the features with features values marginally sampled
    # from the other dataset.
    pad_sw_idx = rng.choice(n_samples_lw, size=n_samples_sw, replace=True)
    X_sw_padded = np.hstack([X_sw, np.take(X_lw, pad_sw_idx, axis=0)])

    pad_lw_idx = rng.choice(n_samples_sw, size=n_samples_lw, replace=True)
    X_lw_padded = np.hstack([np.take(X_sw, pad_lw_idx, axis=0), X_lw])

    # Vertically stack the two datasets and shuffle them.
    X = scale(np.concatenate([X_sw_padded, X_lw_padded], axis=0))
    y = np.concatenate([y_sw, y_lw])
    if y.dtype.kind == "f":
        y = scale(y)

    X = _enforce_estimator_tags_X(est, X)
    y = _enforce_estimator_tags_y(est, y)
    sample_weight = np.concatenate([sample_weight_sw, sample_weight_lw])
    return shuffle(X, y, sample_weight, random_state=rng)
