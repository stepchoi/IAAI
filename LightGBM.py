import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
from load_data_lgbm import load_data
from dateutil.relativedelta import relativedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm
from sklearn.model_selection import train_test_split

space = {
    # better accuracy
    'learning_rate': hp.choice('learning_rate', [0.05, 0.08, 0.1, 0.15]),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']),
    'max_bin': hp.choice('max_bin', [127, 255]),
    'num_leaves': hp.choice('num_leaves', [25, 75, 125, 250, 500]), # np.arange(50, 200, 30, dtype=int)

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', [25, 75, 125, 250]),
    'feature_fraction': hp.choice('feature_fraction', [0.3, 0.5, 0.7, 0.9]),
    'bagging_fraction': hp.choice('bagging_fraction', [0.3, 0.5, 0.7, 0.9]),
    'bagging_freq': hp.choice('bagging_freq', [2, 4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', [0.01, 0.02, 0.05, 0.08]),
    'lambda_l1': hp.choice('lambda_l1', [0, 1, 5, 15]),
    'lambda_l2': hp.choice('lambda_l2', [1, 10, 100, 200]),

    # parameters won't change
    # 'boosting_type': 'gbdt',  # past:  hp.choice('boosting_type', ['gbdt', 'dart']
    'objective': 'regression_l1',     # for regression
    # 'objective': 'multiclass',          # for classification
    'verbose': -1,
    # 'metric': 'multi_error',            # for classification
    'num_threads': 12  # for the best speed, set this to the number of real CPU cores
}

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def lgbm_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    print(params)

    lgb_train = lgb.Dataset(sample_set['train_xx'], label=sample_set['train_yy'], free_raw_data=False)
    lgb_eval = lgb.Dataset(sample_set['valid_x'], label=sample_set['valid_y'], free_raw_data=False, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_eval,
                    num_boost_round=1000,
                    early_stopping_rounds=150)

    # prediction on all sets
    if space['objective'] == 'regression_l1':
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

    return Y_train_pred, Y_valid_pred, Y_test_pred, gbm

def eval(space):
    ''' train & evaluate LightGBM on given space by hyperopt trails '''

    Y_train_pred, Y_valid_pred, Y_test_pred, gbm = lgbm_train(space)
    Y_test = sample_set['test_y']

    result = {  'mae_train': mean_absolute_error(sample_set['train_yy'], Y_train_pred),
                'mae_valid': mean_absolute_error(sample_set['valid_y'], Y_valid_pred),
                'mae_test': mean_absolute_error(Y_test, Y_test_pred),  ##### write Y test
                'r2': r2_score(Y_test, Y_test_pred),
                'status': STATUS_OK}

    sql_result.update(space)        # update hyper-parameter used in model
    sql_result.update(result)       # update result of model
    sql_result['finish_timing'] = dt.datetime.now()

    hpot['all_results'].append(sql_result.copy())
    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']: # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)
        # hpot['best_model'] = gbm
        hpot['best_importance'] = importance_to_sql(gbm)

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
                'r2': r2_score(Y_test, Y_test_pred),
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
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)
        hpot['best_importance'] = importance_to_sql(gbm)

    sql_result['trial_lgbm'] += 1

    return 1 - result['mae_valid']

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''

    hpot['best_mae'] = 1  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()

    if space['objective'] == 'regression_l1':
        best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    elif space['objective'] == 'multiclass':
        hpot['best_mae'] = 0  # record best training (min mae_valid) in each hyperopt
        best = fmin(fn=eval_classify, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(space['objective'], best)

    # write stock_pred for the best hyperopt records to sql
    with engine.connect() as conn:
        pd.DataFrame(hpot['all_results']).to_sql('results_lightgbm', con=conn, index=False, if_exists='append')
        hpot['best_stock_df'].to_sql('results_lightgbm_stock', con=conn, index=False, if_exists='append')
        hpot['best_importance'].to_sql('results_feature_importance', con=conn, index=False, if_exists='append')
    engine.dispose()

    # hpot['best_model'].save_model('models_lgbm/{}_model.txt'.format(sql_result['trial_lgbm']))

    sql_result['trial_hpot'] += 1
    # return best

def to_sql_bins(cut_bins, exist=True):
    ''' write cut_bins & median of each set to DB'''

    if exist == True:
        df = pd.DataFrame(columns=['cut_bins','med_train'])
        df[['cut_bins','med_train']] = df[['cut_bins','med_train']].astype('object')

        for k in cut_bins.keys():     # record cut_bins & median
            df.at[0, k] = cut_bins[k]

        for col in ['qcut_q', 'icb_code', 'testing_period','y_type']:
            df.at[0, col] = sql_result[col]

        df['name'] = sql_result['name']

        with engine.connect() as conn:      # record type of Y
            df.to_sql('results_bins', con=conn, index=False, if_exists='append')
        engine.dispose()
    else:
        print('Already recorded in DB TABLE results_bins')

def pred_to_sql(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    # print('stock-wise prediction: ', df)

    return df

def importance_to_sql(gbm):
    ''' based on gbm model -> records feature importance in DataFrame to be uploaded to DB '''

    df = pd.DataFrame()
    df['name'] = feature_names     # column names
    df['split'] = gbm.feature_importance(importance_type='split')   # split = # of appearance
    df['gain'] = gbm.feature_importance(importance_type='gain')     # gain = total gain

    df = df.set_index('name').T.reset_index(drop=False)
    df.columns = ['importance_type'] + df.columns.to_list()[1:]
    df['trial_lgbm'] = sql_result['trial_lgbm']

    return df

def read_db_last(sql_result, results_table = 'results_lightgbm'):
    ''' read last records on DB TABLE lightgbm_results for resume / trial_no counting '''

    try:
        with engine.connect() as conn:
            db_last = pd.read_sql("SELECT * FROM {} order by finish_timing desc LIMIT 1".format(results_table), conn)
        engine.dispose()

        db_last_param = db_last[['exclude_fwd','icb_code','testing_period']].to_dict('index')[0]
        db_last_trial_hpot = int(db_last['trial_hpot'])
        db_last_trial_lgbm = int(db_last['trial_lgbm'])

        sql_result['trial_hpot'] = db_last_trial_hpot + 1  # trial_hpot = # of Hyperopt performed (n trials each)
        sql_result['trial_lgbm'] = db_last_trial_lgbm + 1  # trial_lgbm = # of Lightgbm performed

    except:
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

    # training / testing sets split params
    indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                   201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    indi_industry_comb = [10, 20, 30, 35, 40, 45, 50, 60, 65]
    indi_industry = [10, 15, 20, 30, 35, 40, 45, 50, 55, 60, 65]
    period_1 = dt.datetime(2013, 3, 31)     # starting point for first testing set
    ''' 502060 is problematic on 2014-9-30, cv 5'''

    # create dict storing values/df used in training
    sql_result = {}     # data write to DB TABLE lightgbm_results
    hpot = {}           # storing data for best trials in each Hyperopt

    # parser
    resume = False      # change to True if want to resume from the last running as on DB TABLE lightgbm_results
    sample_no = 25      # number of training/testing period go over ( 25 = until 2019-3-31)
    sql_result['name'] = 'qcut x - new industry'                     # name = labeling the experiments
    sql_result['qcut_q'] = 10                           # number of Y classes
    sql_result['y_type'] = 'ni'
    use_median = True       # default setting
    chron_valid = False     # default setting

    db_last_param, sql_result = read_db_last(sql_result)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    data = load_data()          # load all data: create load_data.main = df for all samples - within data(CLASS)

    ## ALTER 1: change for classification problem
    # use_median = False
    # sql_result['qcut_q'] = 3
    # space['num_class']= 3,
    # space['objective'] = 'multiclass'
    # space['metric'] = 'multi_error'

    ## ALTER 2: change using chronological last few as validation
    # chron_valid = True

    ## ALTER 3: use eps_ts instead of ni_ts
    # exclude_fwd = False             # False # TRUE = remove fwd_ey, fwd_roic from x (ratios using ibes data)
    # ibes_qcut_as_x = False
    # sql_result['y_type'] = 'ibes'

    ##ALTER 4: use qcut ibes
    exclude_fwd = True
    ibes_qcut_as_x = True


    ''' start roll over testing period(25) / icb_code(16) / cross-validation sets(5) for hyperopt '''

    # for icb_code in indi_models:    # roll over sectors (first 6 icb code)
    #     data.split_icb(icb_code)    # create load_data.sector = samples from specific sectors - within data(CLASS)
    #     sql_result['icb_code'] = icb_code

    for icb_code in indi_industry_comb:   # roll over industries (first 2 icb code)
        data.split_industry(icb_code, combine_ind=True)
        sql_result['icb_code'] = icb_code

        for i in tqdm(range(sample_no)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            # when setting resume = TRUE -> continue training from last records in DB results_lightgbm
            if resume == True:

                if {'icb_code': icb_code, 'testing_period': pd.Timestamp(testing_period),
                    'exclude_fwd': exclude_fwd} == db_last_param:  # if current loop = last records
                    resume = False
                    print('---------> Resume Training', icb_code, testing_period)
                else:
                    print('Not yet resume: params done', icb_code, testing_period)
                    continue

            try:
                sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period,
                                                                                  sql_result['qcut_q'],
                                                                                  sql_result['y_type'],
                                                                                  exclude_fwd=exclude_fwd,
                                                                                  use_median=use_median,
                                                                                  chron_valid=chron_valid,
                                                                                  ibes_qcut_as_x=ibes_qcut_as_x)

                print(feature_names)

                to_sql_bins(cut_bins)   # record cut_bins & median used in Y conversion

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

                    HPOT(space, max_evals=10)   # start hyperopt
                    cv_number += 1

            except:  # if error occurs in hyperopt or lightgbm training : record error to DB TABLE results_error and continue
                pass_error()
                continue

