from hyperopt import hp

space_xgb = {
    'eta': 0.05,
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[6, 8]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    # 'num_leaves': hp.choice('num_leaves', [32, 1024, 2048]),  # remove 75
    'min_child_weight': 0,
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5]), # remove 0.7
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [50, 100]),
    'alpha': 0,
    'lambda': hp.choice('lambda', [1, 3, 5]), # remove 10
    'tree_method': 'exact'
}

space = {}
space[0] = {
    'eta': hp.choice('eta', [0.1, 0.12]),
    'booster': hp.choice('booster', ['gbtree', 'dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 125, 250]),
    'min_child_weight': hp.choice('min_child_weight', [25, 50]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.7, 0.9]),
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': hp.choice('gamma', [0.05, 0.08]),
    'alpha': hp.choice('alpha', [0, 10]),
    'lambda': hp.choice('lambda', [10, 100]),
}

space[11] = {
    'eta': hp.choice('eta', [0.1, 0.5]), # remove 0.12
    'booster': hp.choice('booster', ['dart']),
    'max_depth': hp.choice('max_depth',[8, 15]),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'num_leaves': hp.choice('num_leaves', [125, 250]),  # remove 75
    'min_child_weight': hp.choice('min_child_weight', [1, 5]), # remove 25, 50
    'colsample_bytree': hp.choice('colsample_bytree', [0.9, 1]), # remove 0.7
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': hp.choice('gamma', [0.01, 0.1, 0.5]), # remove 0.08
    'alpha': hp.choice('alpha', [1, 5]),
    'lambda': hp.choice('lambda', [1, 5]), # remove 10
}

space[20] = {
    'eta': hp.choice('eta', [0.12, 0.15]),
    'booster': hp.choice('booster', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [25, 75]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [50, 75]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.3, 0.6]),
    'subsample': hp.choice('subsample', [0.6, 0.8]),
    'gamma': hp.choice('gamma', [0.03, 0.05]),
    'alpha': hp.choice('alpha', [0, 1]),
    'lambda': hp.choice('lambda', [5, 10]),
}

space[30] = {
    'eta': hp.choice('eta', [0.08, 0.1]),
    'booster': hp.choice('booster', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [350, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [15, 25]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.5, 0.6]),
    'subsample': hp.choice('subsample', [0.95, 1]),
    'gamma': hp.choice('gamma', [0.01, 0.03]),
    'alpha': hp.choice('alpha', [2, 3]),
    'lambda': hp.choice('lambda', [10, 20]),
}

space[35] = {
    'eta': hp.choice('eta', [0.08, 0.1]),
    'booster': hp.choice('booster', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [75, 100]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [5, 10]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.4, 0.6]),
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [0.02, 0.03]),
    'alpha': hp.choice('alpha', [1, 3]),
    'lambda': hp.choice('lambda', [1, 5]),
}

space[40] = {
    'eta': hp.choice('eta', [0.05, 0.08]),
    'booster': hp.choice('booster', ['dart']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [50, 100]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [5, 25]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.9, 1]),
    'subsample': hp.choice('subsample', [0.45, 0.6]),
    'gamma': hp.choice('gamma', [0.005, 0.01]),
    'alpha': hp.choice('alpha', [10, 20]),
    'lambda': hp.choice('lambda', [0, 10]),
}

space[45] = {
    'eta': hp.choice('eta', [0.1, 0.15]),
    'booster': hp.choice('booster', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [125, 150]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [50, 60]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.6, 0.7]),
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [0.08, 0.16]),
    'alpha': hp.choice('alpha', [0, 1]),
    'lambda': hp.choice('lambda', [1, 5]), # try 20??
}

space[51] = {
    'eta': hp.choice('eta', [0.03, 0.05]),
    'booster': hp.choice('booster', ['gbtree']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [75, 125]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [15, 25]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.4, 0.5]),
    'subsample': hp.choice('subsample', [0.5, 0.6]),
    'gamma': hp.choice('gamma', [0.2, 0.25]),
    'alpha': hp.choice('alpha', [0, 1]),
    'lambda': hp.choice('lambda', [10, 20]),
}

space[60] = {
    'eta': hp.choice('eta', [0.06, 0.08]),
    'booster': hp.choice('booster', ['gbtree']),
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', [350, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [15, 25]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.7, 0.8]),
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': hp.choice('gamma', [0, 0.01]),
    'alpha': hp.choice('alpha', [8, 10]),
    'lambda': hp.choice('lambda', [50, 100]),
}

space[65] = {
    'eta': hp.choice('eta', [0.01, 0.1]),
    'booster': hp.choice('booster', ['dart']),
    'max_bin': hp.choice('max_bin', [127]),
    'num_leaves': hp.choice('num_leaves', [300, 400]), # np.arange(50, 200, 30, dtype=int)
    'min_child_weight': hp.choice('min_child_weight', [5, 10]),
    'colsample_bynode': hp.choice('colsample_bynode', [0.7, 0.9]),
    'subsample': hp.choice('subsample', [0.7, 0.9]),
    'gamma': hp.choice('gamma', [0, 0.01]),
    'alpha': hp.choice('alpha', [1, 3]),
    'lambda': hp.choice('lambda', [10, 20]),
}

def find_hyperspace(sql_result):

    return space_xgb

    # if sql_result['icb_code'] < 10:
    #     return space[0]
    # elif (sql_result['icb_code'] >= 10) and (sql_result['icb_code'] < 100):
    #     sp = space[sql_result['icb_code']]
    #     if 'mse' in sql_result['name']:
    #         sp.update({'gamma': hp.choice('gamma', [0, 0.001, 0.1]),
    #                    'alpha': hp.choice('alpha', [0, 1, 5]),
    #                    'lambda': hp.choice('lambda', [0, 1, 5]),
    #                    })
    #     return sp
    # elif sql_result['icb_code'] >= 100:
    #     sector_2_ind = {301010: 30, 101020: 11, 201030: 20, 302020: 30, 351020: 35, 502060: 51, 552010: 51, 651010: 65,
    #                     601010: 60, 502050: 51, 101010: 11, 501010: 51, 201020: 20, 502030: 51, 401010: 40, 999999: 0}
    #     print(sector_2_ind[sql_result['icb_code']])
    #     return space[sector_2_ind[sql_result['icb_code']]]

if __name__ == '__main__':
    sql_result = {'icb_code': 999}
    print(find_hyperspace(sql_result))