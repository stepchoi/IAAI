from hyperopt import hp

space_qoq = {}
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

space_qoq[0] = {
    'learning_rate': 0.01,
    'boosting_type': 'dart',
    'max_bin': hp.choice('max_bin', [32, 64]),
    'num_leaves': hp.choice('num_leaves', [125, 250]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 15]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.5, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.9]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 1e-7, 1e-5]),
    'lambda_l1': hp.choice('lambda_l1', [0, 0.1, 100]),
    'lambda_l2': hp.choice('lambda_l2', [10, 25]),
}

space_qoq[11] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.01]),
    'boosting_type': 'dart',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [128, 256]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [16, 64, 128]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.3, 0.5]),
    'bagging_freq': 1,
    'min_gain_to_split': 0,
    'lambda_l1': 0,
    'lambda_l2': hp.choice('lambda_l2', [0, 0.1]),
}

space_qoq[20] = {
    'learning_rate': 0.1,
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [128, 256]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.9]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.1]),
    'lambda_l1': hp.choice('lambda_l1', [1, 5]),
    'lambda_l2': 0
}

space_qoq[30] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.01]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [128, 256]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.9]),
    'bagging_freq': 1,
    'min_gain_to_split': 0,
    'lambda_l1': hp.choice('lambda_l1', [1, 10]),
    'lambda_l2': 0
}

space_qoq[35] = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [32, 64]),
    'num_leaves': hp.choice('num_leaves', [128, 256]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 10]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.1, 0.3]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [1, 5]),
    'lambda_l1': hp.choice('lambda_l1', [1, 0]),
    'lambda_l2': 0
}

space_qoq[40] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [16, 64, 128]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [16, 64, 128]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.5, 0.7]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.1, 1]),
    'lambda_l1': 0,
    'lambda_l2': hp.choice('lambda_l2', [100, 500]),
}

space_qoq[45] = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [16, 32]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [150, 100]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.1, 0.3]),
    'bagging_freq': 1,
    'min_gain_to_split': 0,
    'lambda_l1': hp.choice('lambda_l1', [100, 200]),
    'lambda_l2': 100,
}

space_qoq[51] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.01]),
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [256, 512]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [64, 128]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.9, 0.7]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.1, 1]),
    'lambda_l1': hp.choice('lambda_l1', [1, 10]),
    'lambda_l2': hp.choice('lambda_l2', [100, 50]),
}

space_qoq[60] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.01]),
    'boosting_type': 'dart',
    'max_bin': hp.choice('max_bin', [32, 64]),
    'num_leaves': hp.choice('num_leaves', [256, 128]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [4, 8]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.5, 0.7]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.1, 1]),
    'lambda_l1': hp.choice('lambda_l1', [1, 10]),
    'lambda_l2': hp.choice('lambda_l2', [100, 200]),
}

space_qoq[65] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.01]),
    'boosting_type': 'gbdt',
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [16, 64, 128]),
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25, 50]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.3]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.9, 0.7]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 1]),
    'lambda_l1': 0,
    'lambda_l2': hp.choice('lambda_l2', [50, 100, 500]),
}


# add this one for hyperspace comparison test
space_compare = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': 'dart',
    'max_bin': 128,
    'num_leaves': 125,
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [1, 200]),
    'feature_fraction': hp.choice('feature_fraction', [0.1, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.1, 0.9]),
    'bagging_freq': 1,
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.05, 50]),
    'lambda_l1': hp.choice('lambda_l1', [0, 200]),
    'lambda_l2': 100,
}

space[11] = {
    'learning_rate': hp.choice('learning_rate', [0.08, 0.1]), # remove 0.12
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [125, 250]),  # remove 75
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [10, 15]), # remove 25, 50
    'feature_fraction': hp.choice('feature_fraction', [0.9, 1]), # remove 0.7
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.03]), # remove 0.08
    'lambda_l1': hp.choice('lambda_l1', [15, 20]),
    'lambda_l2': hp.choice('lambda_l2', [30, 50]), # remove 10
}

space[20] = {
    'learning_rate': hp.choice('learning_rate', [0.12, 0.15]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [25, 75]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [50, 75]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.6, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.03, 0.05]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1]),
    'lambda_l2': hp.choice('lambda_l2', [5, 10]),
}

space[30] = {
    'learning_rate': hp.choice('learning_rate', [0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [350, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.5, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.95, 1]),
    'bagging_freq': hp.choice('bagging_freq', [1, 2]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.03]),
    'lambda_l1': hp.choice('lambda_l1', [2, 3]),
    'lambda_l2': hp.choice('lambda_l2', [10, 20]),
}

space[35] = {
    'learning_rate': hp.choice('learning_rate', [0.08, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [75, 100]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 10]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.6]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [12, 14]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.02, 0.03]),
    'lambda_l1': hp.choice('lambda_l1', [1, 3]),
    'lambda_l2': hp.choice('lambda_l2', [1, 5]),
}

space[40] = {
    'learning_rate': hp.choice('learning_rate', [0.05, 0.08]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [50, 100]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.9, 1]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.45, 0.6]),
    'bagging_freq': hp.choice('bagging_freq', [2, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.005, 0.01]),
    'lambda_l1': hp.choice('lambda_l1', [10, 20]),
    'lambda_l2': hp.choice('lambda_l2', [0, 10]),
}

space[45] = {
    'learning_rate': hp.choice('learning_rate', [0.1, 0.15]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [125, 150]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [50, 60]),
    'feature_fraction': hp.choice('feature_fraction', [0.6, 0.7]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.8]),
    'bagging_freq': hp.choice('bagging_freq', [2, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.08, 0.16]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1]),
    'lambda_l2': hp.choice('lambda_l2', [1, 5]), # try 20??
}

space[51] = {
    'learning_rate': hp.choice('learning_rate', [0.03, 0.05]),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 125]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.4, 0.5]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.5, 0.6]),
    'bagging_freq': hp.choice('bagging_freq', [12, 20]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.2, 0.25]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1]),
    'lambda_l2': hp.choice('lambda_l2', [10, 20]),
}

space[60] = {
    'learning_rate': hp.choice('learning_rate', [0.06, 0.08]),
    'boosting_type': hp.choice('boosting_type', ['gbdt']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [350, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [15, 25]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.8]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.8, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 4]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.01]),
    'lambda_l1': hp.choice('lambda_l1', [8, 10]),
    'lambda_l2': hp.choice('lambda_l2', [50, 100]),
}

space[65] = {
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1]),
    'boosting_type': hp.choice('boosting_type', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [300, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [5, 10]),
    'feature_fraction': hp.choice('feature_fraction', [0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.7, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [1, 16]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.01]),
    'lambda_l1': hp.choice('lambda_l1', [1, 3]),
    'lambda_l2': hp.choice('lambda_l2', [10, 20]),
}

def find_hyperspace(sql_result):

    if sql_result['y_type'] == 'ibes_qoq':
        space = space_qoq

    if sql_result['icb_code'] < 10:
        if 'compare' in sql_result['name']:
            return space_compare
        else:
            return space[0]
    elif (sql_result['icb_code'] >= 10) and (sql_result['icb_code'] < 100):
        sp = space[sql_result['icb_code']]
        if 'mse' in sql_result['name']:
            sp.update({'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.001, 0.1]),
                       'lambda_l1': hp.choice('lambda_l1', [0, 1, 5]),
                       'lambda_l2': hp.choice('lambda_l2', [0, 1, 5]),
                       })
        return sp
    elif sql_result['icb_code'] >= 100:
        sector_2_ind = {301010: 30, 101020: 11, 201030: 20, 302020: 30, 351020: 35, 502060: 51, 552010: 51, 651010: 65,
                        601010: 60, 502050: 51, 101010: 11, 501010: 51, 201020: 20, 502030: 51, 401010: 40, 999999: 0}
        print(sector_2_ind[sql_result['icb_code']])
        return space[sector_2_ind[sql_result['icb_code']]]

if __name__ == '__main__':
    sql_result = {'icb_code': 999}
    print(find_hyperspace(sql_result))