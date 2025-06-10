import pytest

import numpy as np
from sklearn import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from sample_weight_audit.data import make_data_for_estimator


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_make_data_for_classifier(seed):
    clf = LogisticRegression(C=1)
    X, y, sample_weight = make_data_for_estimator(
        clf, n_samples=5000, n_features=10, random_state=seed
    )
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0] == sample_weight.shape[0]
    X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = (
        train_test_split(X, y, sample_weight, train_size=2000, random_state=seed)
    )

    # Check that ignoring the weights leads to different coef values.
    clf_with_sw = clone(clf).fit(X_train, y_train, sample_weight=sample_weight_train)
    clf_without_sw = clone(clf).fit(X_train, y_train)

    predictive_coef_magnitude = 0.3
    assert np.abs(clf_with_sw.coef_[:, :5]).max() < predictive_coef_magnitude
    assert np.abs(clf_with_sw.coef_[:, 5:]).max() > predictive_coef_magnitude

    assert np.abs(clf_without_sw.coef_[:, :5]).max() > predictive_coef_magnitude
    assert np.abs(clf_without_sw.coef_[:, 5:]).max() < predictive_coef_magnitude

    assert np.abs(clf_with_sw.coef_ - clf_without_sw.coef_).max() > 0.1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_make_data_for_regressor(seed):
    reg = Ridge()
    X, y, sample_weight = make_data_for_estimator(
        reg, n_samples=5000, n_features=10, random_state=seed
    )
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0] == sample_weight.shape[0]
    X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = (
        train_test_split(X, y, sample_weight, train_size=2000, random_state=seed)
    )

    # Check that ignoring the weights leads to different coef values.
    reg_with_sw = clone(reg).fit(X_train, y_train, sample_weight=sample_weight_train)
    reg_without_sw = clone(reg).fit(X_train, y_train)

    predictive_coef_magnitude = 0.3
    assert np.abs(reg_with_sw.coef_[:5]).max() < predictive_coef_magnitude
    assert np.abs(reg_with_sw.coef_[5:]).max() > predictive_coef_magnitude

    assert np.abs(reg_without_sw.coef_[:5]).max() > predictive_coef_magnitude
    assert np.abs(reg_without_sw.coef_[5:]).max() < predictive_coef_magnitude

    assert np.abs(reg_with_sw.coef_ - reg_without_sw.coef_).max() > 0.1
