from hyperopt import hp
base_space = {'min_child_weight': 0, 'alpha': 0, 'tree_method': 'exact'}

def find_l1_space():
    ''' found space when objective is regression_l1 '''

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
        'booster': 'dart',
        'max_depth': hp.choice('max_depth',[4, 6]),
        'max_bin': hp.choice('max_bin',[128, 256]),
        'colsample_bytree': hp.choice('colsample_bytree',  [0.7, 0.8]), # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.8, 0.9]),
        'gamma': 50,
        'lambda': hp.choice('lambda', [5, 3]),
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
        'lambda': hp.choice('lambda', [10, 5]), # remove 10
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
        'max_depth': hp.choice('max_depth',[6, 8]),
        'max_bin': 64,
        'colsample_bytree': hp.choice('colsample_bytree', [0.3, 0.5, 0.4]), # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.7, 0.8]),
        'gamma': 50,
        'lambda': hp.choice('lambda', [3, 5]), # remove 10
        'eta': 0.05,
    }

    space[65] = {
        'booster': hp.choice('booster',['dart','gbtree']),
        'max_depth': hp.choice('max_depth',[8, 12]),
        'max_bin': hp.choice('max_bin', [64, 128]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.5, 0.9]), # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.2, 0.7]),
        'gamma': hp.choice('gamma', [100, 50]),
        'lambda': hp.choice('lambda', [5, 10, 100]), # remove 10
        'eta': hp.choice('eta', [0.1, 0.05]),
    }

    return space

def find_l2_space():
    ''' found space when objective is regression_l1 '''

    space = {}
    space[0] = {
        'booster': hp.choice('booster',['dart','gbtree']),
        'max_depth': hp.choice('max_depth',[4, 8, 12]),
        'max_bin': hp.choice('max_bin', [64, 128, 512]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.5, 0.8]), # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.2, 0.5, 0.8]),
        'gamma': hp.choice('gamma', [50, 100, 500]),
        'lambda': hp.choice('lambda', [0, 1, 5, 20, 100]), # remove 10
        'eta': hp.choice('eta', [0.05, 0.1, 0.01]),
    }

    space[11] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [8, 10]),
        'max_bin': hp.choice('max_bin', [256, 384]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.6]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.5]),
        'gamma': hp.choice('gamma', [400, 500]),
        'lambda': hp.choice('lambda', [0, 1]),  # remove 10
        'eta': 0.05,
    }

    space[20] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [10, 11]),
        'max_bin': hp.choice('max_bin', [256, 512]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.9]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.9, 1]),
        'gamma': hp.choice('gamma', [400, 500]),
        'lambda': 0,  # remove 10
        'eta': 0.05,
    }

    space[30] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [12]),
        'max_bin': 32,
        'colsample_bytree': hp.choice('colsample_bytree', [0.9, 0.8]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.8, 0.6]),
        'gamma': hp.choice('gamma', [50, 100]),
        'lambda': 0,  # remove 10
        'eta': 0.05,
    }

    space[35] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [10]),
        'max_bin': hp.choice('max_bin', [128, 32, 16]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.4]),  # remove 0.7
        'colsample_bylevel': hp.choice('colsample_bylevel', [0.8]),
        'subsample': hp.choice('subsample', [0.9, 0.8]),
        'gamma': hp.choice('gamma', [300, 400]),
        'lambda': hp.choice('lambda', [0]),  # remove 10
        'eta': 0.05,
    }

    space[40] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [11, 10]),
        'max_bin': hp.choice('max_bin', [512, 1024]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.7, 0.9]),  # remove 0.7
        'colsample_bylevel': hp.choice('colsample_bylevel', [0.8]),
        'subsample': hp.choice('subsample', [0.6, 0.7]),
        'gamma': 500,
        'lambda': hp.choice('lambda', [0]),  # remove 10
        'eta': 0.05,
    }

    space[45] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [7]),
        'max_bin': hp.choice('max_bin', [256, 128]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.5]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.8]),
        'gamma': 100,
        'lambda': hp.choice('lambda', [3, 5]),  # remove 10
        'eta': 0.05
    }

    space[51] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [11, 10, 9]),
        'max_bin': hp.choice('max_bin', [384, 256]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.9]),  # remove 0.7
        'colsample_bylevel': hp.choice('colsample_bylevel', [0.9]),
        'subsample': hp.choice('subsample', [0.7, 0.75]),
        'gamma': 100,
        'lambda': hp.choice('lambda', [1]),  # remove 10
        'eta': 0.05,
    }

    space[60] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [8, 12]),
        'max_bin': 64,
        'colsample_bytree': hp.choice('colsample_bytree', [0.6]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.8]),
        'gamma': 50,
        'lambda': hp.choice('lambda', [3, 6, 7]),  # remove 10
        'eta': 0.05,
    }

    space[65] = {
        'booster': 'gbtree',
        'max_depth': 6,
        'max_bin': 64,
        'colsample_bytree': hp.choice('colsample_bytree', [0.8]),  # remove 0.7
        'colsample_bylevel': hp.choice('colsample_bylevel', [1]),
        'subsample': hp.choice('subsample', [0.5]),
        'gamma': 100,
        'lambda': hp.choice('lambda', [1]),  # remove 10
        'eta': 0.05,
    }

    return space

def find_l2_round_space():
    ''' found space when objective is regression_l1 '''

    space = {}
    space[0] = {
        'booster': hp.choice('booster',['dart','gbtree']),
        'max_depth': hp.choice('max_depth',[4, 8, 12]),
        'max_bin': hp.choice('max_bin', [64, 128, 512]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.5, 0.8]), # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.2, 0.5, 0.8]),
        'gamma': hp.choice('gamma', [50, 100, 500]),
        'lambda': hp.choice('lambda', [0, 1, 5, 20, 100]), # remove 10
        'eta': hp.choice('eta', [0.05, 0.1, 0.01]),
    }

    space[11] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [4, 5, 6]),
        'max_bin': 128,
        'colsample_bytree': hp.choice('colsample_bytree', [0.3, 0.9]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.7, 0.8, 0.9]),
        'gamma': 500,
        'lambda': hp.choice('lambda', [5, 10, 20]),  # remove 10
        'eta': 0.05,
    }

    space[20] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [6, 12]),
        'max_bin': 64,
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.3, 0.4]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.4, 0.5, 0.6]),
        'gamma': hp.choice('gamma', [100, 500]),
        'lambda': hp.choice('lambda', [1, 3, 5]),  # remove 10
        'eta': 0.05,
    }

    space[30] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [10, 12, 15]),
        'max_bin': hp.choice('max_bin', [128, 256]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.7, 0.8, 0.9]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.7, 0.8, 0.9]),
        'gamma': hp.choice('gamma', [100, 500]),
        'lambda': hp.choice('lambda', [1, 10, 20]),  # remove 10
        'eta': 0.05,
    }

    space[35] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [8, 10]),
        'max_bin': hp.choice('max_bin', [512, 256]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.4, 0.5, 0.6]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.6, 0.8]),
        'gamma': 100,
        'lambda': hp.choice('lambda', [0, 10, 20]),  # remove 10
        'eta': 0.1,
    }

    space[40] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [12, 10]),
        'max_bin': 128,
        'colsample_bytree': hp.choice('colsample_bytree', [0.7, 0.8, 0.9]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.7, 0.8, 0.9]),
        'gamma': hp.choice('lambda', [500, 100]),
        'lambda': hp.choice('lambda', [0, 10, 20]),  # remove 10
        'eta': hp.choice('lambda', [0.1, 0.05])
    }

    space[45] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [4, 6]),
        'max_bin': hp.choice('max_bin', [64, 128]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.3, 0.4]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.5, 0.7]),
        'gamma': 500,
        'lambda': hp.choice('lambda', [1, 10, 20]),  # remove 10
        'eta': 0.05,
    }

    space[51] = {
        'booster': 'gbtree',
        'max_depth': hp.choice('max_depth', [8, 10]),
        'max_bin': hp.choice('max_bin', [256, 512]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.6, 0.8]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.5, 0.7]),
        'gamma': hp.choice('gamma', [50, 100]),
        'lambda': hp.choice('lambda', [5, 10, 20]),  # remove 10
        'eta': 0.05,
    }

    space[60] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [8, 10]),
        'max_bin': 64,
        'colsample_bytree': hp.choice('colsample_bytree', [0.7, 0.8, 0.9]), # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.5, 0.7]),
        'gamma': 500,
        'lambda': hp.choice('lambda', [0, 100]),  # remove 10
        'eta': hp.choice('eta', [0.01, 0.1]),
    }

    space[65] = {
        'booster': 'dart',
        'max_depth': hp.choice('max_depth', [4, 6]),
        'max_bin': hp.choice('max_bin', [128, 256]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.2, 0.4]),  # remove 0.7
        'colsample_bylevel': 1,
        'subsample': hp.choice('subsample', [0.3, 0.9]),
        'gamma': 100,
        'lambda': hp.choice('lambda', [1, 100]),  # remove 10
        'eta': 0.1,
    }

    return space

def find_hyperspace(sql_result):

    if sql_result['objective'] == 'mae':
        space = find_l1_space()
    elif sql_result['objective'] == 'rmse':
        if sql_result['qcut_q'] == 0:
            space = find_l2_round_space()
        else:
            space = find_l2_space()

    myspace = space[sql_result['icb_code']]
    myspace.update(base_space)
    return myspace

if __name__ == '__main__':
    sql_result = {'icb_code': 999}
    print(find_hyperspace(sql_result))