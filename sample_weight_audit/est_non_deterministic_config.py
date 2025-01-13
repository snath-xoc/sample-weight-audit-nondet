
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

def get_config():
    return NON_DEFAULT_PARAMS
    
