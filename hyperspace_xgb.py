from hyperopt import hp
base_space = {'min_child_weight': 0, 'alpha': 0, 'tree_method': 'exact'}

space = {}
space[0] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[6, 8]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [50, 100]),
    'lambda': hp.choice('lambda', [1, 3, 5]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.1]),
}

space[11] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[4, 5]),
    'max_bin': hp.choice('max_bin', [16, 32]),
    'colsample_bytree': 1,
    'colsample_bylevel': hp.choice('colsample_bylevel', [0.4, 0.8, 0.2]), # remove 0.7
    'subsample': hp.choice('subsample', [0.4, 0.5]),
    'gamma': hp.choice('gamma', [500, 600]),
    'lambda': hp.choice('lambda', [3, 4]), # remove 10
    'eta': 0.05,
}

space[20] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[2, 8, 12]),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.8, 0.2]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.4, 0.8]),
    'gamma': hp.choice('gamma', [100, 500]),
    'lambda': hp.choice('lambda', [0, 1]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.1]),
}

space[30] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[4, 8, 12]),
    'max_bin': hp.choice('max_bin', [256, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.7]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': 10,
    'lambda': hp.choice('lambda', [3, 30, 300]),
    'eta': 0.05,
}

space[35] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[10, 8]),
    'max_bin': 256,
    'colsample_bytree': hp.choice('colsample_bytree', [0.7, 0.8]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': hp.choice('gamma', [1000, 10000]),
    'lambda': hp.choice('lambda', [20, 50, 100]), # remove 10
    'eta': 0.05,
}

space[40] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[6, 10]),
    'max_bin': 64,
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.2]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.2, 0.7]),
    'gamma': hp.choice('gamma', [500, 100]),
    'lambda': hp.choice('gamma', [10, 5]), # remove 10
    'eta': 0.05,
}

space[45] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[5, 6]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.3]), # remove 0.7
    'colsample_bylevel': hp.choice('colsample_bylevel', [0.8, 0.5]), # remove 0.7
    'subsample': 0.7,
    'gamma': hp.choice('gamma', [100, 500]),
    'lambda': hp.choice('lambda', [0, 1]), # remove 10
    'eta': 0.05,
}

space[51] = {
    'booster': 'gbtree',
    'max_depth': hp.choice('max_depth',[4, 5]),
    'max_bin': 128,
    'colsample_bytree': hp.choice('colsample_bytree', [0.9, 0.7]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.2, 0.7]),
    'gamma': hp.choice('gamma', [100, 500]),
    'lambda': hp.choice('lambda', [0, 1e-3]), # remove 10
    'eta':0.05,
}

space[60] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[8, 12]),
    'max_bin': 64,
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.6]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.7, 0.2]),
    'gamma': hp.choice('eta', [25, 50]),
    'lambda': hp.choice('lambda', [3, 30, 300]), # remove 10
    'eta': hp.choice('eta', [0.1, 0.05]),
}

space[65] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[8, 12]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.9]), # remove 0.7
    'colsample_bylevel': 1,
    'subsample': hp.choice('subsample', [0.2, 0.7]),
    'gamma': hp.choice('eta', [100, 50]),
    'lambda': hp.choice('lambda', [5, 10, 100]), # remove 10
    'eta': hp.choice('eta', [0.1, 0.05]),
}

def find_hyperspace(sql_result):

    if sql_result['icb_code'] < 10:
        space[0].update(base_space)
        return space[0]
    elif (sql_result['icb_code'] >= 10) and (sql_result['icb_code'] < 100):
        sp = space[sql_result['icb_code']]
        sp.update(base_space)
        return sp
    elif sql_result['icb_code'] >= 100:
        sector_2_ind = {301010: 30, 101020: 11, 201030: 20, 302020: 30, 351020: 35, 502060: 51, 552010: 51, 651010: 65,
                        601010: 60, 502050: 51, 101010: 11, 501010: 51, 201020: 20, 502030: 51, 401010: 40, 999999: 0}
        space[sector_2_ind[sql_result['icb_code']]].update(base_space)
        return space[sector_2_ind[sql_result['icb_code']]]

if __name__ == '__main__':
    sql_result = {'icb_code': 999}
    print(find_hyperspace(sql_result))