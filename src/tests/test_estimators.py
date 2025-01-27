import numpy as np
import sys

sys.path.insert(1, "../")
from scipy.stats import multinomial, kstest
import pytest

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
    RegressorMixin,
    clone,
    is_regressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from sample_weight_audit import weighted_repeated_fit_equivalence_test


class NoisyClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        classifier=None,
        temperature=2.0,
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

        base_clf = clone(classifier).fit(X, y, sample_weight=sample_weight)

        y_proba = base_clf.predict_proba(X)

        # Scale by inverse temperature to control amount of injected stochasticity:
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
        noise_scale=2,
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
        self._random_offsets = rng.normal(scale=self.noise_scale, size=X.shape[1])
        return self

    def transform(self, X):
        return self._base_transformer.transform(X) + self._random_offsets


class NoisyRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        regressor=None,
        noise_scale=2,
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
        return self

    def predict(self, X):
        rng = check_random_state(self.random_state)
        random_offsets = rng.normal(scale=self.noise_scale, size=X.shape[0])
        return self._base_regressor.predict(X) + random_offsets



@pytest.mark.parametrize("ignore_sample_weight", [False, True])
@pytest.mark.parametrize("test", ["welch", "kstest", "ed_perm", "mannwhitneyu"])
@pytest.mark.parametrize(
    "est", [NoisyClassifier(), NoisyRegressor(), NoisyTransformer()],
    ids=["classifier", "regressor", "transformer"],
)
def test_equivalence_on_noisy_estimator(est, test, ignore_sample_weight):
    est = est
    if is_regressor(est):
        max_seed = 30
    else:
        max_seed = 10

    results = weighted_repeated_fit_equivalence_test(
        est,
        n_features=10,
        test=test,
        max_seed=max_seed,
        train_size=300,
        n_samples_per_cv_group=500,
        rep_test_size=20,
        max_repeats=5,
        n_classes=4,
        correct_threshold=True,
        equal_var=False,
        issparse=False,
        ignore_sample_weight=ignore_sample_weight,
    )

    if ignore_sample_weight:
        assert results["min_p_value"] < 0.05
    else:
        pvalues = np.sort(results["p_values"].flatten())
        uniform = np.sort(np.random.uniform(size=pvalues.shape[0]))
        assert (kstest(uniform, pvalues).pvalue) > 0.05
