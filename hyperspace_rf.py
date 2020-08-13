from hyperopt import hp

space_rf = {
    'max_depth': hp.choice('max_depth', [6, 8, 12, 15]),
    'min_samples_split': hp.choice('min_samples_split', [1, 25, 100]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [25, 50, 200]),
    'min_weight_fraction_leaf': hp.choice('min_weight_fraction_leaf', [0, 0.5, 1]),
    'max_features': hp.choice('max_features', [0.5, 0.8]),
    'min_impurity_decrease': hp.choice('min_impurity_decrease', [0, 1e-3, 0.1]),  # remove 0.08
    'max_samples': hp.choice('max_samples', [0.6, 0.8]),

    # 'max_leaf_nodes': hp.choice('max_leaf_nodes', []),
    'n_jobs': 12,
    'bootstrap': True}


def find_hyperspace(sql_result):

    return space_rf
