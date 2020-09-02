import datetime as dt
import xgboost as xgb
import numpy as np
import argparse
import pandas as pd
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm
import matplotlib.pyplot as plt

from load_data_lgbm import load_data
from hyperspace_xgb import find_hyperspace


def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    params['gamma'] = params['eta']/params['gamma']  # convert gamma_multiple to gamma
    print(params)

    lgb_train = xgb.DMatrix(sample_set['train_xx'], label=sample_set['train_yy'])
    lgb_eval = xgb.DMatrix(sample_set['valid_x'], label=sample_set['valid_y'])

    def huber_approx_obj(preds, dtrain):
        d = preds - dtrain.get_label()  # remove .get_labels() for sklearn
        h = 1  # h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess

    evals_result = {}

    if args.objective == 'mae':
        gbm = xgb.train(params=params,
                        dtrain=lgb_train,
                        evals=[(lgb_eval,'valid'), (lgb_train,'train')],
                        evals_result=evals_result,
                        num_boost_round=400,
                        early_stopping_rounds=50,
                        obj=huber_approx_obj)
    elif args.objective == 'rmse':
        gbm = xgb.train(params=params,
                        dtrain=lgb_train,
                        evals=[(lgb_eval, 'valid'), (lgb_train, 'train')],
                        evals_result=evals_result,
                        num_boost_round=400,
                        early_stopping_rounds=50)
    else:
        NameError('WRONG objective in arguments - use [mae, rmse] instead')

    # prediction on all sets
    Y_train_pred = gbm.predict(xgb.DMatrix(sample_set['train_xx']))
    Y_valid_pred = gbm.predict(xgb.DMatrix(sample_set['valid_x']))
    Y_test_pred = gbm.predict(xgb.DMatrix(sample_set['test_x']))

    return Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, gbm


def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trials '''

    Y_train_pred, Y_valid_pred, Y_test_pred, evals_result, gbm = lgbm_train(space)
    Y_test = sample_set['test_y']

    result = {'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
              'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
              'mae_test': mean_absolute_error(Y_test, Y_test_pred),  ##### write Y test
              'mse_train': mean_squared_error(sample_set['train_yy'], Y_train_pred),
              'mse_valid': mean_squared_error(sample_set['valid_y'], Y_valid_pred),
              'mse_test': mean_squared_error(Y_test, Y_test_pred),  ##### write Y test
              'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
              'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
              'r2_test': r2_score(Y_test, Y_test_pred),
              'status': STATUS_OK}

    sql_result.update(space)  # update hyper-parameter used in model
    sql_result.update(result)  # update result of model
    sql_result['finish_timing'] = dt.datetime.now()

    hpot['all_results'].append(sql_result.copy())
    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_plot'] = evals_result
        hpot['best_model'] = gbm
        hpot['best_trial'] = sql_result['trial_lgbm']

    sql_result['trial_lgbm'] += 1

    return result['mae_valid']

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['best_mae'] = 1  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()

    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)

    # write stock_pred for the best hyperopt records to sql
    with engine.connect() as conn:
        hpot['best_stock_df'].to_sql('results_xgboost_stock', con=conn, index=False, if_exists='append', method='multi')
        pd.DataFrame(hpot['all_results']).to_sql('results_xgboost', con=conn, index=False, if_exists='append', method='multi')
    engine.dispose()

    sql_result['trial_hpot'] += 1

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    df['name'] = sql_result['name']
    # print('stock-wise prediction: ', df)

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_sql', required=True)
    parser.add_argument('--objective', default='mae')
    parser.add_argument('--exclude_stock', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--exclude_fwd', default=False, action='store_true')
    parser.add_argument('--sample_type', default='industry')
    parser.add_argument('--y_type', default='ibes')
    parser.add_argument('--sample_no', type=int, default=21)
    parser.add_argument('--qcut_q', default=10, type=int)
    parser.add_argument('--trial_lgbm_add', default=1, type=int)
    parser.add_argument('--sample_ratio', default=1, type=float)
    parser.add_argument('--nthread', default=12, type=int)
    args = parser.parse_args()

    # training / testing sets split par
    market_list = ['normal']  # default setting = all samples cross countries
    if args.sample_type == 'industry':
        partitions = [11, 20, 30, 35, 40, 45, 51, 60, 65] # industry partition
    elif args.sample_type == 'sector':
        partitions = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                      201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    elif args.sample_type == 'entire':
        partitions = [0]
    else:
        NameError('Wrong sample_type in arguments')

    period_1 = dt.datetime(2013, 3, 31)  # starting point for first testing set
    base_space = {'verbosity': 0,
                  'nthread': args.nthread,
                  'eval_metric': args.objective,
                  'grow_policy':'depthwise'}

    # create dict storing values/df used in training
    sql_result = {}  # data write to DB TABLE lightgbm_results
    hpot = {}  # storing data cxe  best trials in each Hyperopt
    resume = args.resume  # change to True if want to resume from the last running as on DB TABLE lightgbm_results
    sample_no = args.sample_no  # number of training/testing period go over ( 25 = until 2019-3-31)

    load_data_params = {'exclude_fwd': args.exclude_fwd,
                        'use_median': True,
                        'chron_valid': False,
                        'y_type': 'ibes',
                        'qcut_q': 10,
                        'ibes_qcut_as_x': not (args.exclude_fwd),
                        'exclude_stock': args.exclude_stock}

    for mkt in market_list:  # roll over partition for each market (for IIIb)
        data = load_data(macro_monthly=True, market=mkt, sample_ratio=args.sample_ratio)
        sql_result['objective'] = args.objective

        x_type_map = {True: 'fwdepsqcut', False: 'ni'}  # True/False based on exclude_fwd
        sql_result['x_type'] = x_type_map[args.exclude_fwd]
        sql_result['name'] = args.name_sql  # label experiment

        # update load_data data
        sql_result['qcut_q'] = load_data_params['qcut_q']  # number of Y classes
        sql_result['y_type'] = load_data_params['y_type']

        ''' start roll over testing period(25) / icb_code(9) / cross-validation sets(5) for hyperopt '''
        for icb_code in partitions:  # roll over industries (first 2 icb code)

            data.split_industry(icb_code, combine_ind=True)
            sql_result['icb_code'] = icb_code

            for i in tqdm(range(sample_no)):  # roll over testing period
                testing_period = period_1 + i * relativedelta(months=3)
                sql_result['testing_period'] = testing_period

                sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, **load_data_params)
                sql_result['exclude_fwd'] = load_data_params['exclude_fwd']

                space = find_hyperspace(sql_result)
                space.update(base_space)

                cv_number = 1  # represent which cross-validation sets
                for train_index, valid_index in cv:  # roll over 5 cross validation set
                    sql_result['cv_number'] = cv_number

                    # when Resume = False: try split validation set from training set + start hyperopt
                    sample_set['valid_x'] = sample_set['train_x'][valid_index]
                    sample_set['train_xx'] = sample_set['train_x'][train_index]  # train_x is in fact train & valid set
                    sample_set['valid_y'] = sample_set['train_y'][valid_index]
                    sample_set['train_yy'] = sample_set['train_y'][train_index]

                    sql_result['train_len'] = len(sample_set['train_xx'])  # record length of training/validation sets
                    sql_result['valid_len'] = len(sample_set['valid_x'])

                    try:
                        HPOT(space, max_evals=10)  # start hyperopt
                    except:
                        pass

                    cv_number += 1

