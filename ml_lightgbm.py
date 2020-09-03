import datetime as dt
import lightgbm as lgb
import argparse
import pandas as pd

from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm

from hyperspace_lgbm import find_hyperspace
from load_data_lgbm import load_data

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()

    lgb_train = lgb.Dataset(sample_set['train_xx'], label=sample_set['train_yy'], free_raw_data=False)
    lgb_eval = lgb.Dataset(sample_set['valid_x'], label=sample_set['valid_y'], free_raw_data=False, reference=lgb_train)

    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_eval, lgb_train],
                    valid_names=['valid', 'train'],
                    num_boost_round=1000,
                    early_stopping_rounds=150,
                    feature_name = feature_names,
                    evals_result=evals_result)

    # prediction on all sets
    Y_train_pred = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)
    Y_valid_pred = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
    Y_test_pred = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)

    return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, gbm

def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials '''

    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, gbm = lgbm_train(space)
    Y_test = sample_set['test_y']

    result = {  'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
                'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
                'mae_test': mean_absolute_error(Y_test, Y_test_pred),  ##### write Y test
                'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
                'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
                'mse_test': mean_squared_error(Y_test, Y_test_pred),  ##### write Y test
                'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
                'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
                'r2_test': r2_score(Y_test, Y_test_pred),
                'status': STATUS_OK}

    if args.objective == 'regression_l2':
        return result['mse_valid']
    elif args.objective == 'regression_l1':
        return result['mae_valid']
    else:
        NameError('Objective not evaluated!')

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    return best

if __name__ == "__main__":

    parser = argparse.ArgumentParser()      # setting for different configuration
    parser.add_argument('--objective', default='regression_l1')     # regression_l2 = optimizing mse
    parser.add_argument('--exclude_fwd', default=False, action='store_true')        # True = without ibes as X
    parser.add_argument('--sample_type', default='industry')        # sampling type
    parser.add_argument('--y_type', default='ibes')         # ibes_qoq for qoq type growth prediction
    parser.add_argument('--qcut_q', default=10, type=int)
    parser.add_argument('--sample_ratio', default=1, type=float)    # 0.5 = select 50% random sample from original data set
    parser.add_argument('--nthread', default=12, type=int)
    args = parser.parse_args()

    # training / testing sets split par
    if args.sample_type == 'industry':
        partitions = [11, 20, 30, 35, 40, 45, 51, 60, 65]   # 11 represents industry (10 + 15); 51 represents (50 + 55)
    elif args.sample_type == 'entire':
        partitions = [0]    # 0 represents aggregate model

    period_1 = dt.datetime(2013, 3, 31)     # starting point for first testing set
    base_space = {'verbose': -1,
                  'num_threads': args.nthread}  # for the best speed, set this to the number of real CPU cores

    load_data_params = {'exclude_fwd': args.exclude_fwd,
                        'y_type': args.y_type,
                        'qcut_q': args.qcut_q,
                        }

    data = load_data(sample_ratio=args.sample_ratio)          # load data step 1

    for icb_code in partitions:   # roll over industries (first 2 icb code)
        data.split_industry(icb_code)      # load data step 2

        for i in tqdm(range(21)):  # roll over 2013-3-31 to 2018-3-31
            testing_period = period_1 + i * relativedelta(months=3)
            sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, **load_data_params)    # load data step 3

            for train_index, valid_index in cv:     # roll over 5 cross validation set
                sample_set['valid_x'] = sample_set['train_x'][valid_index]
                sample_set['train_xx'] = sample_set['train_x'][train_index] # train_x is in fact train & valid set
                sample_set['valid_y'] = sample_set['train_y'][valid_index]
                sample_set['train_yy'] = sample_set['train_y'][train_index]

                space = find_hyperspace(args)
                space.update(base_space)
                HPOT(space, max_evals=10)   # start hyperopt



