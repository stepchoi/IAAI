from hyperopt import hp

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
    'learning_rate': hp.choice('learning_rate', [0.08, 0.1]), # remove 0.12
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [125, 250]),  # remove 75
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 10, 15]), # remove 25, 50
    'feature_fraction': hp.choice('feature_fraction', [0.8, 0.9]), # remove 0.7
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.03, 0.05]), # remove 0.08
    'lambda_l1': hp.choice('lambda_l1', [5, 10, 15]),
    'lambda_l2': hp.choice('lambda_l2', [30, 50, 70]), # remove 10
}

space[20] = {
    'learning_rate': hp.choice('learning_rate', [0.12, 0.15]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [25, 75]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [75, 100, 125]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.4, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.03, 0.05]),
    'lambda_l1': hp.choice('lambda_l1', [0, 0.1, 0.5]),
    'lambda_l2': hp.choice('lambda_l2', [5, 10, 15]),
}

space[30] = {
    'learning_rate': hp.choice('learning_rate', [0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [350, 500, 600]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.6, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.9, 0.95]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.03]),
    'lambda_l1': hp.choice('lambda_l1', [3, 5]),
    'lambda_l2': hp.choice('lambda_l2', [10, 20]),
}

space[35] = {
    'learning_rate': hp.choice('learning_rate', [0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [75, 100]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 10, 15]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.5, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.7, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [12, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.02, 0.03]),
    'lambda_l1': hp.choice('lambda_l1', [1, 3]),
    'lambda_l2': hp.choice('lambda_l2', [1, 5]),
}

space[40] = {
    'learning_rate': hp.choice('learning_rate', [0.06, 0.08, 0.5]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [50, 200, 500]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 50, 100]),
    'feature_fraction': hp.choice('feature_fraction', [0.8, 1]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.3, 0.6]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2, 32]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.01, 0.1]),
    'lambda_l1': hp.choice('lambda_l1', [5, 10, 100]),
    'lambda_l2': hp.choice('lambda_l2', [0, 5, 10]),
}

space[45] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.15]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [125, 150]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [50, 60]),
    'feature_fraction': hp.choice('feature_fraction', [0.6, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.7]),
    'bagging_freq': hp.choice('bagging_freq', [4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.16, 0.2]),
    'lambda_l1': hp.choice('lambda_l1', [0, 0.1, 0.5]),
    'lambda_l2': hp.choice('lambda_l2', [1, 5, 20]), # try 20??
}

space[51] = {
    'learning_rate': hp.choice('learning_rate', [0.03, 0.05, 0.07]),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 125]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.5]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.5, 0.6]),
    'bagging_freq': hp.choice('bagging_freq', [12, 16, 20]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.16, 0.2]),
    'lambda_l1': hp.choice('lambda_l1', [0, 0.1, 0.5]),
    'lambda_l2': hp.choice('lambda_l2', [10, 20]),
}

space[60] = {
    'learning_rate': hp.choice('learning_rate', [0.06, 0.08, 0.5]),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [300, 350, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.8, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.9, 1]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.01]),
    'lambda_l1': hp.choice('lambda_l1', [10, 15]),
    'lambda_l2': hp.choice('lambda_l2', [50, 100, 200]),
}

space[65] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [300, 350, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [1, 5, 10]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [1, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.01]),
    'lambda_l1': hp.choice('lambda_l1', [0.5, 1]),
    'lambda_l2': hp.choice('lambda_l2', [1, 20]),
}

def find_hyperspace(sql_result):

    if sql_result['icb_code'] < 10:
        return space[0]
    elif sql_result['icb_code'] >= 10:
        try:
            return space[sql_result['icb_code']]
        except:
            return space[0]


