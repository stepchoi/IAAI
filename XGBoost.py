import datetime as dt

import xgboost as xgb
import numpy as np
import argparse
import pandas as pd
from load_data_lgbm import load_data
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, mean_squared_error
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from hyperspace_lgbm import find_hyperspace


parser = argparse.ArgumentParser()
parser.add_argument('--sp_only', default=False, action='store_true')
parser.add_argument('--exclude_stock', default=False, action='store_true')
args = parser.parse_args()


base_space ={'objective': 'regression_l1',     # for regression
            'verbose': -1,
            'num_threads': 12}  # for the best speed, set this to the number of real CPU cores

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    print(params)

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

    # plot_history(evals_result, gbm, sql_result['trial_lgbm'])

    # prediction on all sets
    if space['objective'] in ['regression_l1', 'regression_l2']:
        Y_train_pred = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)
        Y_valid_pred = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
        Y_test_pred = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)

    elif space['objective'] == 'multiclass':
        Y_train_pred_softmax = gbm.predict(sample_set['train_xx'], num_iteration=gbm.best_iteration)
        Y_train_pred = [list(i).index(max(i)) for i in Y_train_pred_softmax]
        Y_valid_pred_softmax = gbm.predict(sample_set['valid_x'], num_iteration=gbm.best_iteration)
        Y_valid_pred = [list(i).index(max(i)) for i in Y_valid_pred_softmax]
        Y_test_pred_softmax = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)
        Y_test_pred = [list(i).index(max(i)) for i in Y_test_pred_softmax]

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

    sql_result.update(space)        # update hyper-parameter used in model
    sql_result.update(result)       # update result of model
    sql_result['finish_timing'] = dt.datetime.now()

    hpot['all_results'].append(sql_result.copy())
    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']: # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_plot'] = evals_result
        hpot['best_model'] = gbm
        hpot['best_trial'] = sql_result['trial_lgbm']
        hpot['best_importance'] = to_sql_importance(gbm)

    sql_result['trial_lgbm'] += 1

    return result['mae_valid']

def eval_classify(space):
    ''' train & evaluate LightGBM on given space by hyperopt trails '''

    Y_train_pred, Y_valid_pred, Y_test_pred, gbm = lgbm_train(space)
    Y_test = sample_set['test_y']

    result = {
                'mae_train': accuracy_score(sample_set['train_yy'], Y_train_pred),   # use column names of regression
                'mae_valid': accuracy_score(sample_set['valid_y'], Y_valid_pred),
                'mae_test': accuracy_score(Y_test, Y_test_pred),
                'r2_train': r2_score(sample_set['train_yy'], Y_train_pred),
                'r2_valid': r2_score(sample_set['valid_y'], Y_valid_pred),
                'r2_test': r2_score(Y_test, Y_test_pred),
                'status': STATUS_OK}

    sql_result.update(space)        # update hyper-parameter used in model
    sql_result.update(result)       # update result of model
    sql_result.pop('num_class')
    sql_result.pop('metric')
    sql_result['finish_timing'] = dt.datetime.now()

    hpot['all_results'].append(sql_result.copy())
    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] > hpot['best_mae']: # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = to_sql_prediction(Y_test_pred)
        hpot['best_importance'] = to_sql_importance(gbm)

    sql_result['trial_lgbm'] += 1

    return 1 - result['mae_valid']

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['best_mae'] = 1  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()

    if space['objective'] in ['regression_l1', 'regression_l2']:
        best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    elif space['objective'] == 'multiclass':
        hpot['best_mae'] = 0  # record best training (min mae_valid) in each hyperopt
        best = fmin(fn=eval_classify, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(space['objective'], best)

    # write stock_pred for the best hyperopt records to sql
    with engine.connect() as conn:
        hpot['best_stock_df'].to_sql('results_lightgbm_stock', con=conn, index=False, if_exists='append', method='multi')
        hpot['best_importance'].to_sql('results_feature_importance', con=conn, index=False, if_exists='append', method='multi')
        pd.DataFrame(hpot['all_results']).to_sql('results_lightgbm', con=conn, index=False, if_exists='append', method='multi')
    engine.dispose()

    plot_history(hpot['best_plot'], hpot['best_model'], hpot['best_trial'])

    if sql_result['icb_code']==201030:
        hpot['best_model'].save_model('models_lgbm/model_201030_{}.txt'.format(hpot['best_trial']))

    sql_result['trial_hpot'] += 1
    # return best

def plot_history(evals_result, gbm, trial_num):
    ''' plot the training loss history '''

    try:
        ax = lgb.plot_metric(evals_result, metric='l1')     # plot training / validation loss
    except:
        ax = lgb.plot_metric(evals_result, metric='l2')


    print(sql_result['icb_code'], sql_result['testing_period'].strftime('%Y-%m'))
    ax.set_title('{}:{}:{}'.format(sql_result['icb_code'], sql_result['testing_period'].strftime('%Y-%m'),
                                   sql_result['x_type']))
    fig = ax.get_figure()
    fig.savefig('models_lgbm/plot_lgbm_eval_{}.png'.format(trial_num))
    plt.close()

    if sql_result['icb_code']==201030:
        ax = lgb.plot_importance(gbm, max_num_features=20)    # plot feature importance
        fig = ax.get_figure()
        fig.savefig('models_lgbm/plot_lgbm_impt_{}.png'.format(trial_num))
        plt.close()

def to_sql_bins(cut_bins, write=True):
    ''' write cut_bins & median of each set to DB'''

    if write == False:

        df = pd.DataFrame(columns=['cut_bins','med_train'])
        df[['cut_bins','med_train']] = df[['cut_bins','med_train']].astype('object')

        for k in cut_bins.keys():     # record cut_bins & median
            df.at[0, k] = cut_bins[k]

        for col in ['qcut_q', 'icb_code', 'testing_period','y_type']:
            df.at[0, col] = sql_result[col]

        with engine.connect() as conn:      # record type of Y
            df.to_sql('results_bins_new', con=conn, index=False, if_exists='append')
        engine.dispose()

def to_sql_prediction(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    # print('stock-wise prediction: ', df)

    return df

def to_sql_importance(gbm):
    ''' based on gbm model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    df['name'] = feature_names     # column names
    df['split'] = gbm.feature_importance(importance_type='split')   # split = # of appearance
    df['gain'] = gbm.feature_importance(importance_type='gain')     # gain = total gain

    df = df.set_index('name').T.reset_index(drop=False)
    df.columns = ['importance_type'] + df.columns.to_list()[1:]
    df['trial_lgbm'] = sql_result['trial_lgbm']

    return df

def read_db_last(sql_result, results_table = 'results_lightgbm', first=False):
    ''' read last records on DB TABLE lightgbm_results for resume / trial_no counting '''

    if first == False:
        with engine.connect() as conn:
            db_last = pd.read_sql("SELECT * FROM {} Order by finish_timing desc LIMIT 1".format(results_table), conn)
        engine.dispose()

        db_last_param = db_last[['icb_code','testing_period']].to_dict('index')[0]
        db_last_trial_hpot = int(db_last['trial_hpot'])
        db_last_trial_lgbm = int(db_last['trial_lgbm'])

        sql_result['trial_hpot'] = db_last_trial_hpot + 1  # trial_hpot = # of Hyperopt performed (n trials each)
        sql_result['trial_lgbm'] = db_last_trial_lgbm + 1  # trial_lgbm = # of Lightgbm performed
        print('if resume from: ', db_last_param,'; sql last trial_lgbm: ', sql_result['trial_lgbm'])
    else:
        db_last_param = None
        sql_result['trial_hpot'] = sql_result['trial_lgbm'] = 0

    return db_last_param, sql_result

def pass_error():
    ''' continue loop when encounter error in trials '''

    print('ERROR on', sql_result)

    with engine.connect() as conn:
        pd.DataFrame({'icb_code': sql_result['icb_code'],
                      'testing_period': pd.Timestamp(sql_result['testing_period']),
                      'qcut_q': sql_result['qcut_q'],
                      'recording_time': dt.datetime.now()}, index=[0],).to_sql('results_error', con=conn,
                                                                         index=False, if_exists='append')
    engine.dispose()

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    # training / testing sets split par
    indi_sector = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                   201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    indi_industry_new = [11, 20, 30, 35, 40, 45, 51, 60, 65]
    indi_industry = [10, 15, 20, 30, 35, 40, 45, 50, 55, 60, 65]
    period_1 = dt.datetime(2013, 3, 31)     # starting point for first testing set

    # create dict storing values/df used in training
    sql_result = {}     # data write to DB TABLE lightgbm_results
    hpot = {}           # storing data for best trials in each Hyperopt
    load_data_params = {'exclude_fwd': True,
                        'use_median': True,
                        'chron_valid': False,
                        'y_type': 'ibes',
                        'qcut_q': 10,
                        'ibes_qcut_as_x': False,
                        'exclude_stock': args.exclude_stock}

    # default parser
    macro_monthly = True # remember to change main.csv
    resume = False      # change to True if want to resume from the last running as on DB TABLE lightgbm_results
    sample_no = 25      # number of training/testing period go over ( 25 = until 2019-3-31)

    data = load_data(macro_monthly=macro_monthly, sp_only=args.sp_only)          # load all data: create load_data.main = df for all samples - within data(CLASS)

    # FINAL 1: use ibes_y + without ibes data
    # sql_result['objective'] = base_space['objective'] = 'regression_l2'
    sql_result['x_type'] = 'fwdepsqcut'

    # update load_data data
    sql_result['qcut_q'] = load_data_params['qcut_q']     # number of Y classes
    sql_result['y_type'] = load_data_params['y_type']

    ''' start roll over testing period(25) / icb_code(16) / cross-validation sets(5) for hyperopt '''

    db_last_param, sql_result = read_db_last(sql_result)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    for icb_code in indi_industry_new:   # roll over industries (first 2 icb code)

        sql_result['name'] = 'ibes_industry -sp500'  # name = labeling the experiments

        data.split_industry(icb_code, combine_ind=True)
        sql_result['icb_code'] = icb_code

        for i in tqdm(range(sample_no)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            # when setting resume = TRUE -> continue training from last records in DB results_lightgbm
            if resume == True:

                if {'icb_code': icb_code, 'testing_period': pd.Timestamp(testing_period)} == db_last_param:  # if current loop = last records
                    resume = False
                    print('---------> Resume Training', icb_code, testing_period)
                else:
                    print('Not yet resume: params done', icb_code, testing_period)
                    continue

            # if sample_no==25:
            try:
                sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, **load_data_params)
                sql_result['exclude_fwd'] = load_data_params['exclude_fwd']

                # print('23355L106' in test_id)
                print(feature_names)

                # to_sql_bins(cut_bins)   # record cut_bins & median used in Y conversion

                cv_number = 1   # represent which cross-validation sets
                for train_index, valid_index in cv:     # roll over 5 cross validation set
                    sql_result['cv_number'] = cv_number

                    # when Resume = False: try split validation set from training set + start hyperopt
                    sample_set['valid_x'] = sample_set['train_x'][valid_index]
                    sample_set['train_xx'] = sample_set['train_x'][train_index] # train_x is in fact train & valid set
                    sample_set['valid_y'] = sample_set['train_y'][valid_index]
                    sample_set['train_yy'] = sample_set['train_y'][train_index]

                    sql_result['train_len'] = len(sample_set['train_xx']) # record length of training/validation sets
                    sql_result['valid_len'] = len(sample_set['valid_x'])

                    space = find_hyperspace(sql_result)
                    space.update(base_space)
                    print(space)

                    HPOT(space, max_evals=10)   # start hyperopt
                    cv_number += 1

                # exit(0)

            except:  # if error occurs in hyperopt or lightgbm training : record error to DB TABLE results_error and continue
                # exit(0)
                pass_error()
                cv_number += 1
                continue

