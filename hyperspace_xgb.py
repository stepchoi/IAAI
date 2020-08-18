from hyperopt import hp
base_space = {'min_child_weight': 0, 'alpha': 0, 'tree_method': 'exact'}

space = {}
space[0] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[6, 8]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5]), # remove 0.7
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [50, 100]),
    'lambda': hp.choice('lambda', [1, 3, 5]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.1]),
}

space[11] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[5, 6, 8]),
    'max_bin': hp.choice('max_bin', [32, 64]),
    'colsample_bylevel': hp.choice('colsample_bylevel', [0.4, 0.5]), # remove 0.7
    'subsample': hp.choice('subsample', [0.6, 0.7]),
    'gamma': hp.choice('gamma', [100, 500]),
    'lambda': hp.choice('lambda', [3, 5]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.01]),
}

space[20] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[6, 8]),
    'max_bin': hp.choice('max_bin', [32, 64]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5]), # remove 0.7
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [10, 50]),
    'lambda': hp.choice('lambda', [2, 3]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.1]),
}

space[30] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[5, 6, 8]),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.7, 0.9]), # remove 0.7
    'subsample': hp.choice('subsample', [0.6, 0.7]),
    'gamma': hp.choice('gamma', [10, 50]),
    'lambda': hp.choice('lambda', [2, 3]), # remove 10
    'eta': hp.choice('eta', [0.1, 0.5])
}

space[40] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[8, 10]),
    'max_bin': hp.choice('max_bin', [32, 64]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.6]), # remove 0.7
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': hp.choice('gamma', [50, 100]),
    'lambda': hp.choice('lambda', [5, 10]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.01]),
}

space[45] = {
    'booster': hp.choice('booster',['dart','gbtree']),
    'max_depth': hp.choice('max_depth',[6, 8]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5]), # remove 0.7
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [100, 500]),
    'lambda': hp.choice('lambda', [1, 3, 5]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.01]),
}

space[51] = {
    'booster': 'gbtree',
    'max_depth': hp.choice('max_depth',[5, 6]),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.7]), # remove 0.7
    'subsample': hp.choice('subsample', [0.6, 0.7]),
    'gamma': hp.choice('gamma', [100, 500]),
    'lambda': hp.choice('lambda', [0, 1]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.01]),
}

space[60] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[8, 10]),
    'max_bin': hp.choice('max_bin', [32, 64]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.7]), # remove 0.7
    'subsample': hp.choice('subsample', [0.7, 0.8]),
    'gamma': hp.choice('gamma', [10, 50]),
    'lambda': hp.choice('lambda', [3, 4]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.1]),
}

space[65] = {
    'booster': 'dart',
    'max_depth': hp.choice('max_depth',[5, 6]),
    'max_bin': hp.choice('max_bin', [64, 128]),
    'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5]), # remove 0.7
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'gamma': hp.choice('gamma', [10, 50]),
    'lambda': hp.choice('lambda', [1, 3]), # remove 10
    'eta': hp.choice('eta', [0.05, 0.01]),
}

def find_hyperspace(sql_result):

    if sql_result['icb_code'] < 10:
        return space[0].update(base_space)
    elif (sql_result['icb_code'] >= 10) and (sql_result['icb_code'] < 100):
        sp = space[sql_result['icb_code']]
        return sp.update(base_space)
    elif sql_result['icb_code'] >= 100:
        sector_2_ind = {301010: 30, 101020: 11, 201030: 20, 302020: 30, 351020: 35, 502060: 51, 552010: 51, 651010: 65,
                        601010: 60, 502050: 51, 101010: 11, 501010: 51, 201020: 20, 502030: 51, 401010: 40, 999999: 0}
        print(sector_2_ind[sql_result['icb_code']])
        return space[sector_2_ind[sql_result['icb_code']]].update(base_space)

if __name__ == '__main__':
    sql_result = {'icb_code': 999}
    print(find_hyperspace(sql_result))