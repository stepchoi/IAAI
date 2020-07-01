import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from load_data import load_data
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, explained_variance_score, max_error, mean_squared_error, \
    mean_squared_log_error, median_absolute_error, r2_score
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm
from sklearn.model_selection import train_test_split

space = {
    # better accuracy
    'learning_rate': hp.choice('learning_rate', [0.01, 0.1, 0.5]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [127, 255]),
    'num_leaves': hp.choice('num_leaves', [25, 75, 125, 250]), # np.arange(50, 200, 30, dtype=int)

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 75, 125, 250]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.5, 0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.3, 0.5, 0.7, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0, 0.1, 0.02, dtype='d')),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 5, 15]),
    'lambda_l2': hp.choice('lambda_l2', [1, 10, 100, 500]),

    # parameters won't change
    # 'boosting_type': 'gbdt',  # past:  hp.choice('boosting_type', ['gbdt', 'dart']
    'objective': 'regression_l1',
    'verbose': -1,
    # 'metric': 'multi_error',
    'num_threads': 12  # for the best speed, set this to the number of real CPU cores
}


def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    print(params)

    lgb_train = lgb.Dataset(X_train, label=Y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_valid, label=Y_valid, free_raw_data=False, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=1000,
                    early_stopping_rounds=150,
                    )

    # gbm.save_model('model.txt')

    # prediction on all sets
    Y_train_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    Y_valid_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    Y_test_pred = gbm.predict(sample_set['test_x'], num_iteration=gbm.best_iteration)

    return Y_train_pred, Y_valid_pred, Y_test_pred

def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trails '''

    Y_train_pred, Y_valid_pred, Y_test_pred = lgbm_train(space)
    Y_test = sample_set['test_ni']

    result = {  'mae_train': mean_absolute_error(Y_train, Y_train_pred),
                'mae_valid': mean_absolute_error(Y_valid, Y_valid_pred),
                'mae_test': mean_absolute_error(Y_test, Y_test_pred),  ##### write Y test
                # 'mae_train_org': mean_absolute_error(Y_train, Y_train_pred),
                # 'mae_test_org': mean_absolute_error(Y_test, Y_test_pred),
                'status': STATUS_OK}

    sql_result.update(space)        # update hyper-parameter used in model
    sql_result.update(result)       # update result of model
    sql_result['finish_timing'] = dt.datetime.now()

    pt = pd.DataFrame.from_records([sql_result], index=[0])
    print('sql_result_before writing: ', sql_result)

    with engine.connect() as conn:
        pt.to_sql('results_lightgbm', con=conn, index=False, if_exists='append')
    engine.dispose()

    if result['mae_valid'] < best_mae: # update best_mae to the lowest value for Hyperopt
        best_mae = result['mae_valid']
        best_stock_df = pred_to_sql(Y_test_pred)

    sql_result['trial_lgbm'] += 1

    return result['mae_valid']

def pred_to_sql(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    print('stock-wise prediction: ', df)

    return df

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''
    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)

    # write stock_pred for the best hyperopt records to sql
    with engine.connect() as conn:
        best_stock_df.to_sql('results_lightgbm_stock', con=conn, index=False, if_exists='append')
    engine.dispose()

    sql_result['trial_hpot'] += 1
    # return best

def to_sql_bins(cut_bins):
    ''' write cut_bins & median of each set to DB'''

    with engine.connect() as conn:      # record type of Y
        exist = pd.read_sql("SELECT * FROM results_bins WHERE qcut_q={} AND icb_code={} AND testing_period='{}'".format(
            qcut_q, icb_code, str(testing_period)), con=conn)
    engine.dispose()

    if len(exist) < 1: # if db has not records med_train / cut_bin for trial yet

        df = pd.DataFrame(columns=['cut_bins','med_train','med_test'])
        df[['cut_bins','med_train','med_test']] = df[['cut_bins','med_train','med_test']].astype('object')

        for k in cut_bins['ni'].keys():     # record cut_bins & median
            df.at[0, k] = cut_bins['ni'][k]

        for col in ['qcut_q', 'icb_code', 'testing_period']:
            df.at[0, col] = sql_result[col]

        with engine.connect() as conn:      # record type of Y
            df.to_sql('results_bins', con=conn, index=False, if_exists='append')
        engine.dispose()
    else:
        print('Already recorded in DB TABLE results_bins')

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    try:    # read last records on DB TABLE lightgbm_results for resume / trial_no counting
        db_last = pd.read_sql("SELECT * FROM results_lightgbm order by finish_timing desc LIMIT 1", engine)  # identify current # trials from past execution
        db_last_trial_hpot = int(db_last['trial_hpot'])
        db_last_trial_lgbm = int(db_last['trial_lgbm'])
    except:
        db_last_trial_hpot = 0
        db_last_trial_lgbm = 0

    indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                   201020, 502030, 401010, 'miscel']  # icb_code with > 1300 samples + rests in single big model

    # parser
    resume = False      # change to True if want to resume from the last running as on DB TABLE lightgbm_results
    qcut_q = 10         # number of Y classes
    sample_no = 25      # number of training/testing period go over ( 25 = until 2019-3-31)
    ''' DEBUG: change to 28 for official run '''

    # records params to be written to DB
    sql_result = {}                                 # sql_result
    sql_result['name'] = 'change y to med_train'                    # name = labeling the experiments
    sql_result['trial_hpot'] = db_last_trial_hpot + 1  # trial_hpot = # of Hyperopt performed (n trials each)
    sql_result['trial_lgbm'] = db_last_trial_lgbm + 1  # trial_lgbm = # of Lightgbm performed
    sql_result['qcut_q'] = qcut_q

    data = load_data()      # load all data: create load_data.main = df for all samples - within data(CLASS)

    period_1 = dt.datetime(2013, 3, 31)

    for icb_code in indi_models:    # roll over industries

        data.split_icb(icb_code)    # create load_data.sector = samples from specific sectors - within data(CLASS)
        print('icb_code: ', icb_code)
        sql_result['icb_code'] = icb_code

        for i in tqdm(range(sample_no)):  # roll over testing period

            testing_period = period_1 + i * relativedelta(months=3)
            print('testing_period: ', testing_period)
            sql_result['testing_period'] = testing_period

            sample_set, cut_bins, cv, test_id = data.split_all(testing_period, qcut_q)   # split train / valid / test

            to_sql_bins(cut_bins)   # record cut_bins & median used in Y conversion

            cv_number = 1   # represent which cross-validation sets
            for train_index, valid_index in cv:     # roll over 5 cross validation set
                sql_result['cv_number'] = cv_number

                X_train, X_valid = sample_set['train_x'][train_index], sample_set['train_x'][valid_index]
                Y_train, Y_valid = sample_set['train_ni'][train_index], sample_set['train_ni'][valid_index]  # lightGBM use Net Income as Y
                print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

                best_mae = 1
                best_stock_df = pd.DataFrame()
                HPOT(space, max_evals=10)
                cv_number += 1
                # exit(0)
