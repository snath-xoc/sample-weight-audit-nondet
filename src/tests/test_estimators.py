import numpy as np

from scipy.stats import multinomial
import pytest

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
    RegressorMixin,
    clone,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sample_weight_audit import check_weighted_repeated_estimator_fit_equivalence


class NoisyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        classifier=None,
        temperature=0.1,
        random_state=None,
        ignore_sample_weight=False,
    ):
        self.classifier = classifier
        self.random_state = random_state
        self.ignore_sample_weight = ignore_sample_weight
        self.temperature = temperature

    def fit(self, X, y, sample_weight=None):
        rng = check_random_state(self.random_state)
        if self.classifier is None:
            classifier = LogisticRegression()
        else:
            classifier = self.classifier

        if self.ignore_sample_weight:
            sample_weight = None

        # Let's first fit a classifier that is assumed to respect sample weights
        # deterministically.
        base_clf = clone(classifier).fit(X, y, sample_weight=sample_weight)
        y_proba = base_clf.predict_proba(X)

        # Make the fit non-deterministic by sampling stochastic labels from the
        # probabilities returned by the deterministic classifier: scale by
        # inverse temperature to control amount of injected stochasticity.
        y_proba /= self.temperature
        y_proba /= y_proba.sum(axis=1, keepdims=True)
        noisy_y = np.concatenate(
            [
                multinomial.rvs(n=1, p=p, size=1, random_state=rng).argmax(axis=1)
                for p in y_proba
            ]
        )
        self._noisy_clf = clone(classifier).fit(X, noisy_y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self._noisy_clf.predict_proba(X)


class NoisyTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        transformer=None,
        noise_scale=0.01,
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
        self._random_offsets = rng.normal(scale=self.noise_scale, size=X_trans.shape[1]) * np.std(
            X_trans, axis=0
        )
        return self

    def transform(self, X):
        return self._base_transformer.transform(X) + self._random_offsets


class NoisyRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        regressor=None,
        noise_scale=0.01,
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


@pytest.mark.parametrize("ignore_sample_weight", [False, True])
@pytest.mark.parametrize("test_name", ["welch", "kstest", "mannwhitneyu", "ed_perm"])
@pytest.mark.parametrize(
    "est",
    [
        NoisyClassifier(random_state=0),
        NoisyRegressor(random_state=0),
        NoisyTransformer(random_state=0),
    ],
    ids=["noisy_classifier", "noisy_regressor", "noisy_transformer"],
)
def test_equivalence_on_noisy_estimator(est, test_name, ignore_sample_weight):
    results = check_weighted_repeated_estimator_fit_equivalence(
        est, test_name=test_name, random_state=0
    )
    if ignore_sample_weight:
        assert results.min_p_value < 0.05
    else:
        assert results.mean_p_value > 0.05
