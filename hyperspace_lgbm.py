from hyperopt import hp

def find_space_l1():
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

    space['us'] = {}
    space['hk'] = {}
    space['jp'] = {}
    space['cn'] = {}

    return space

def find_space_l2():

    space = {}
    space[11] = {
        'learning_rate': 0.05,
        'boosting_type': 'dart',
        'max_bin': 255,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 100, 150, 10),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 50, 10),
        'feature_fraction': hp.quniform('feature_fraction', 0.7, 0.9, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.7, 0.9, 0.1),
        'bagging_freq': 4,
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 10, 15, 5),
    }

    space[20] = {
        'learning_rate': 0.1,
        'boosting_type': 'dart',
        'max_bin': 255,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 50, 200, 50),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 20, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.4, 0.8, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.8, 0.9, 0.1),
        'bagging_freq': 4,
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 0, 5, 1),
    }

    space[30] = {
        'learning_rate': 0.1,
        'boosting_type': 'dart',
        'max_bin': 255,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 150, 200, 10),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 30, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.7, 0.9, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.4, 0.8, 0.1),
        'bagging_freq': hp.quniform('bagging_freq', 4, 8, 2),
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 15, 20, 1),
    }

    space[35] = {
        'learning_rate': 0.05,
        'boosting_type': 'dart',
        'max_bin': 127,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 50, 150, 20),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 40, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.6, 0.9, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.7, 0.8, 0.1),
        'bagging_freq': hp.choice('bagging_freq', [2, 8]),
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 0, 15, 5),
    }

    space[40] = {
        'learning_rate': 0.05,
        'boosting_type': 'dart',
        'max_bin': 255,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 50, 150, 20),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 30, 50, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.5, 0.7, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 0.6, 0.1),
        'bagging_freq': 6,
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 0, 10, 2),
    }

    space[45] = {
        'learning_rate': 0.1,
        'boosting_type': 'dart',
        'max_bin': 127,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 50, 150, 20),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 40, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.6, 0.8, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.4, 0.6, 0.1),
        'bagging_freq': 6,
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 0, 10, 2),
    }

    space[51] = {
        'learning_rate': 0.03,
        'boosting_type': 'gbdt',
        'max_bin': 255,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 50, 100, 10),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 20, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.7, 0.9, 0.2),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.2, 0.5, 0.1),
        'bagging_freq': hp.quniform('bagging_freq', 2, 8, 2),
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 0, 20, 5),
    }

    space[60] = {
        'learning_rate': 0.1,
        'boosting_type': 'gbdt',
        'max_bin': 255,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 100, 200, 20),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 5, 20, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.3, 0.5, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.3, 0.5, 0.1),
        'bagging_freq': hp.quniform('bagging_freq', 2, 8, 2),
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 10, 20, 2),
    }

    space[65] = {
        'learning_rate': 0.1,
        'boosting_type': 'dart',
        'max_bin': 127,
        'min_gain_to_split': 0,
        'num_leaves': hp.quniform('num_leaves', 50, 150, 20),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 30, 5),
        'feature_fraction': hp.quniform('feature_fraction', 0.6, 0.8, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.4, 0.6, 0.1),
        'bagging_freq': hp.quniform('bagging_freq', 4, 8, 2),
        'lambda_l1': 0,
        'lambda_l2': hp.quniform('lambda_l2', 0, 5, 1),
    }

    return space

def find_hyperspace(args):
    ''' find hyperspace for given industry/model'''
    base_space = {'verbose': -1, 'num_threads': args.nthread}

    if args.objective == 'regression_l1':
        space = find_space_l1()
    elif args.objective == 'regression_l2':
        space = find_space_l2()

    myspace = space[args.icb_code]
    myspace.update(base_space)
    return myspace
