from hyperopt import hp
import numpy as np

space = {}
space[0] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.12]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 125, 250]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [1, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.05, 0.08]),
    'lambda_l1': hp.choice('lambda_l1', [0, 10]),
    'lambda_l2': hp.choice('lambda_l2', [10, 100]),
}

space[11] = {
    'learning_rate': hp.choice('learning_rate', [0.08, 0.12]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 125, 250]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.05, 0.08]),
    'lambda_l1': hp.choice('lambda_l1', [0, 10]),
    'lambda_l2': hp.choice('lambda_l2', [10, 50]),
}

space[20] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.12]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [25, 75]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.5, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [1, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.05, 0.08]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 3]),
    'lambda_l2': hp.choice('lambda_l2', [10, 50]),
}

space[30] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [125, 250, 500]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 75]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.5, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [1, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.03, 0.05]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 3]),
    'lambda_l2': hp.choice('lambda_l2', [20, 50]),
}

space[35] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.12]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [25, 50, 75]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [2, 8, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.02, 0.05]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1]),
    'lambda_l2': hp.choice('lambda_l2', [1, 20, 50]),
}

space[40] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.08, 0.12]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [50, 75, 100]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 10, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.8]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [2, 8, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.005, 0.01, 0.02]),
    'lambda_l1': hp.choice('lambda_l1', [10, 20]),
    'lambda_l2': hp.choice('lambda_l2', [10, 50]), # try 20??
}

space[45] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [75, 125]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.8]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [8, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.08, 0.12, 0.16]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 5]),
    'lambda_l2': hp.choice('lambda_l2', [1, 20, 100]), # try 20??
}

space[51] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.05, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 250]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 75]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.7]),
    'bagging_freq': hp.choice('bagging_freq', [2, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.08, 0.12, 0.16]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 3]),
    'lambda_l2': hp.choice('lambda_l2', [1, 10, 100]), # try 20??
}

space[60] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 350, 500]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.05]),
    'lambda_l1': hp.choice('lambda_l1', [5, 8, 10]),
    'lambda_l2': hp.choice('lambda_l2', [1, 100]), # try 20??
}

space[65] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.12]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [75, 350, 500]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 30, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.7]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.02]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 10]),
    'lambda_l2': hp.choice('lambda_l2', [1, 50]), # try 20??
}

def find_hyperspace(sql_result):

    if sql_result['icb_code'] < 10:
        return space[0]
    elif sql_result['icb_code'] >= 10:
        try:
            return space[sql_result['icb_code']]
        except:
            return space[0]


