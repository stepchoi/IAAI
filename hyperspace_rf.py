from hyperopt import hp
import numpy as np

space_rf = {
    'n_estimators': 100,
    'max_depth': hp.choice('max_depth',[8, 12]),
    'min_samples_split': hp.choice('min_samples_split',[25, 50, 100]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [25, 50, 100]),
    'min_weight_fraction_leaf': 0.1,
    'max_features': hp.choice('max_features',[0.7, 0.8]),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', [0, 1e-3]),
    'max_samples': hp.choice('max_samples',[0.5, 0.9]),

    'n_jobs': 2}


def find_hyperspace(sql_result):

    return space_rf
