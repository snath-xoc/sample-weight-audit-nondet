import numpy as np

from scipy.special import softmax
import pytest

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
    RegressorMixin,
    clone,
)
from sklearn.utils.estimator_checks import (
    check_sample_weight_equivalence_on_dense_data,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sample_weight_audit import check_weighted_repeated_estimator_fit_equivalence


class NoisyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        classifier=None,
        noise_scale=1e-1,
        random_state=None,
        ignore_sample_weight=False,
    ):
        self.classifier = classifier
        self.random_state = random_state
        self.ignore_sample_weight = ignore_sample_weight
        self.noise_scale = noise_scale

    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)
        if self.classifier is None:
            classifier = LogisticRegression(tol=1e-3, max_iter=1_000)
        else:
            classifier = self.classifier

        if self.ignore_sample_weight:
            sample_weight = None

        # Let's first fit a classifier that is assumed to respect sample weights
        # deterministically.
        self._base_classifier = clone(classifier).fit(X, y, sample_weight=sample_weight)
        y_proba = self._base_classifier.predict_proba(X)

        logits = np.log(y_proba + 1e-12)
        self.logits_offset_ = rng.normal(
            scale=self.noise_scale * np.std(logits, axis=0), size=logits.shape[1]
        )
        return self

    def predict_proba(self, X):
        logits = np.log(self._base_classifier.predict_proba(X) + 1e-12)
        return softmax(logits + self.logits_offset_, axis=1)


class NoisyTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        transformer=None,
        noise_scale=1e-1,
        ignore_sample_weight=False,
        random_state=None,
    ):
        self.transformer = transformer
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.ignore_sample_weight = ignore_sample_weight

    def fit(self, X, y=None, sample_weight=None):
        rng = check_random_state(self.random_state)

        if self.ignore_sample_weight:
            sample_weight = None

        if self.transformer is None:
            transformer = StandardScaler()
        else:
            transformer = clone(self.transformer)

        self._base_transformer = transformer.fit(X, y, sample_weight=sample_weight)
        X_trans = self._base_transformer.transform(X)

        # Make the fit non-deterministic by sampling random offsets to be added to
        # the transformed data:
        self._random_offsets = rng.normal(
            scale=self.noise_scale, size=X_trans.shape[1]
        ) * np.std(X_trans, axis=0)
        return self

    def transform(self, X):
        return self._base_transformer.transform(X) + self._random_offsets


class NoisyRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        regressor=None,
        noise_scale=1e-1,
        ignore_sample_weight=False,
        random_state=None,
    ):
        self.regressor = regressor
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.ignore_sample_weight = ignore_sample_weight

    def fit(self, X, y, sample_weight=None):
        if self.ignore_sample_weight:
            sample_weight = None

        if self.regressor is None:
            regressor = Ridge()
        else:
            regressor = clone(self.regressor)

        self._base_regressor = regressor.fit(X, y, sample_weight=sample_weight)
        y_pred = self._base_regressor.predict(X)

        # Make the fit non-deterministic by sampling a random offset to be
        # added to the predictions:
        rng = check_random_state(self.random_state)
        self.random_offset_ = rng.normal(scale=self.noise_scale) * np.std(y_pred)
        return self

    def predict(self, X):
        return self._base_regressor.predict(X) + self.random_offset_


@pytest.mark.parametrize(
    "test_name",
    [
        "kstest",
        "mannwhitneyu",
    ],
)
@pytest.mark.parametrize(
    "est",
    [
        NoisyClassifier(random_state=0),
        NoisyRegressor(random_state=0),
        NoisyTransformer(random_state=0),
    ],
    ids=["noisy_classifier", "noisy_regressor", "noisy_transformer"],
)
def test_equivalence_on_noisy_estimator(est, test_name):
    good_est = clone(est)
    good_est_result = check_weighted_repeated_estimator_fit_equivalence(
        good_est, test_name=test_name, random_state=0
    )
    bad_est = clone(est).set_params(ignore_sample_weight=True)
    bad_est_result = check_weighted_repeated_estimator_fit_equivalence(
        bad_est, test_name=test_name, random_state=0
    )
    assert bad_est_result.pvalue < good_est_result.pvalue
    assert bad_est_result.pvalue < 0.001
    assert good_est_result.pvalue > 0.01


@pytest.mark.parametrize(
    "est",
    [
        NoisyClassifier(random_state=0, noise_scale=0, ignore_sample_weight=False),
        NoisyRegressor(random_state=0, noise_scale=0, ignore_sample_weight=False),
        NoisyTransformer(random_state=0, noise_scale=0, ignore_sample_weight=False),
    ],
    ids=["noisy_classifier", "noisy_regressor", "noisy_transformer"],
)
def test_equivalence_on_determinitic_estimator(est):
    est_name = est.__class__.__name__
    check_sample_weight_equivalence_on_dense_data(est_name, est)

    bad_est = clone(est).set_params(ignore_sample_weight=True)
    with pytest.raises(AssertionError):
        check_sample_weight_equivalence_on_dense_data(est_name, bad_est)
