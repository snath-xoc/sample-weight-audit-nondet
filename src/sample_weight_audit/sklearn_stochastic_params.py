# List scikit-learn estimator param
from sklearn.calibration import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeClassifier,
)
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC, LinearSVR, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
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
        "estimator": DecisionTreeClassifier(max_depth=1, max_features=0.5)
    },
    AdaBoostRegressor: {
        "estimator": DecisionTreeRegressor(max_depth=1, max_features=0.5)
    },
    LinearSVR: {"dual": True},
    LinearSVC: {"dual": True},
    Ridge: {"solver": "sag"},
    Lasso: {"selection": "random"},
    LassoCV: {"selection": "random"},
    ElasticNet: {"selection": "random"},
    ElasticNetCV: {"selection": "random"},
    DecisionTreeRegressor: {"max_features": 0.5},
    GradientBoostingClassifier: {"max_features": 0.5},
    GradientBoostingRegressor: {"max_features": 0.5},
    HistGradientBoostingRegressor: {"max_features": 0.5},
    HistGradientBoostingClassifier: {"max_features": 0.5},
    RandomForestRegressor: {"max_features": 0.5},
    DecisionTreeClassifier: {"max_features": 0.5},
    DummyClassifier: {"strategy": "stratified"},
    LogisticRegression: {
        "dual": True,
        "solver": "liblinear",
        "max_iter": 10000,
    },
    LogisticRegressionCV: {
        "dual": True,
        "solver": "liblinear",
        "max_iter": 10000,
    },
    NuSVC: {"probability": True},
    RidgeClassifier: {"solver": "saga"},
    SVC: {"probability": True},
    KBinsDiscretizer: {
        "subsample": 50,
        "encode": "ordinal",
        "strategy": "quantile",
    },
    MiniBatchKMeans: {"reassignment_ratio": 0.9},
    RandomTreesEmbedding: {
        "n_estimators": 10,
    },
}
