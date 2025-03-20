# List scikit-learn estimator param
from sklearn.calibration import LinearSVC
from sklearn.cluster import BisectingKMeans, KMeans, MiniBatchKMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    LogisticRegressionCV,
    Perceptron,
    Ridge,
    RidgeClassifier,
)
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVR, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from sklearn.ensemble import GradientBoostingRegressor

# XXX: the values could also be turned into a list in case there are multiple
# known configurations that make the estimator stochastic.

# Parametrizations of scikit-learn estimators that are known to make them stochastic.
STOCHASTIC_FIT_PARAMS = {
    AdaBoostClassifier: {
        "estimator": DecisionTreeClassifier(
            max_features=0.5, min_weight_fraction_leaf=0.1
        )
    },
    AdaBoostRegressor: {
        "estimator": DecisionTreeRegressor(
            max_features=0.5, min_weight_fraction_leaf=0.1
        )
    },
    BaggingClassifier: {"estimator": LogisticRegression()},
    BaggingRegressor: {"estimator": Ridge()},
    BisectingKMeans: {"n_clusters": 10},
    LinearSVR: {"dual": True},
    LinearSVC: {"dual": True},
    Ridge: {"solver": "sag", "max_iter": 100_000},
    Lasso: {"selection": "random"},
    LassoCV: {"selection": "random"},
    ElasticNet: {"selection": "random"},
    ElasticNetCV: {"selection": "random"},
    DecisionTreeClassifier: {"max_features": 0.5, "min_weight_fraction_leaf": 0.1},
    DecisionTreeRegressor: {"max_features": 0.5},
    GradientBoostingClassifier: {"max_features": 0.5},
    GradientBoostingRegressor: {"max_features": 0.5},
    HistGradientBoostingRegressor: {"max_features": 0.5},
    HistGradientBoostingClassifier: {"max_features": 0.5},
    KMeans: {"n_clusters": 10},
    RandomForestRegressor: {"max_features": 0.5},
    DummyClassifier: {"strategy": "stratified"},
    LogisticRegression: {
        "dual": True,
        "solver": "liblinear",
        "max_iter": 100_000,
    },
    LogisticRegressionCV: {
        "solver": "saga",
        "max_iter": 100_000,
    },
    NuSVC: {"probability": True},
    RidgeClassifier: {"solver": "saga", "max_iter": 100_000},
    SVC: {"probability": True},
    KBinsDiscretizer: {
        "subsample": 50,
        "encode": "ordinal",
        "strategy": "quantile",
        "quantile_method": "averaged_inverted_cdf",
    },
    MiniBatchKMeans: {"n_clusters": 10, "reassignment_ratio": 0.01},
    RandomTreesEmbedding: {"n_estimators": 10},
    SGDClassifier: {"max_iter": 100_000},
    SGDRegressor: {"max_iter": 100_000},
    Perceptron: {"max_iter": 100_000},
}
