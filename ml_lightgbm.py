import datetime as dt
import lightgbm as lgb
import argparse

from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm

from hyperspace_lgbm import find_hyperspace
from load_data_lgbm import load_data

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()

    lgb_train = lgb.Dataset(X_train, label=Y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, label=Y_valid, free_raw_data=False, reference=lgb_train)

    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=[lgb_eval, lgb_train],
                    valid_names=['valid', 'train'],
                    num_boost_round=1000,
                    early_stopping_rounds=150,
                    feature_name=feature_names,
                    evals_result=evals_result)

    # prediction on all sets
    Y_train_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_valid_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result

def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials '''

    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result = lgbm_train(space)

    result = {  'mae_train': mean_absolute_error(Y_train, Y_train_pred),
                'mae_valid': mean_absolute_error(Y_valid, Y_valid_pred),
                'mae_test': mean_absolute_error(Y_test, Y_test_pred),
                'mse_train': mean_squared_error(Y_train, Y_train_pred),
                'mse_valid': mean_squared_error(Y_valid, Y_valid_pred),
                'mse_test': mean_squared_error(Y_test, Y_test_pred),
                'r2_train': r2_score(Y_train, Y_train_pred),
                'r2_valid': r2_score(Y_valid, Y_valid_pred),
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
    parser.add_argument('--exclude_fwd', default=False, action='store_true')     # True = no I/B/E/S consensus features
    parser.add_argument('--sample_type', default='industry')
    parser.add_argument('--sample_ratio', default=1, type=float) # 0.5 = select 50% random sample from original data set
    args = parser.parse_args()

    if args.sample_type == 'industry':  # config III
        partitions = [11, 20, 30, 35, 40, 45, 51, 60, 65]   # 11 represents industry (10 + 15); 51 represents (50 + 55)
    elif args.sample_type == 'entire':  # config II
        partitions = [0]    # 0 represents aggregate model

    period_1 = dt.datetime(2013, 4, 1)     # starting point for first testing set

    for icb_code in partitions:   # roll over industries (first 2 icb code)
        for i in tqdm(range(21)):  # roll over 2013-3-31 to 2018-3-31
            testing_period = period_1 + i * relativedelta(months=3) - relativedelta(days=1)
            train_x, train_y, X_test, Y_test, cv, feature_names = load_data(testing_period, **vars(args))

            for train_index, test_index in cv:     # roll over 5 cross validation set
                X_train = train_x[train_index]
                Y_train = train_y[train_index]
                X_valid = train_x[test_index]
                Y_valid = train_y[test_index]

                space = find_hyperspace(args)
                HPOT(space, max_evals=10)   # start hyperopt



