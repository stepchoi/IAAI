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
    'learning_rate': hp.choice('learning_rate', np.arange(0.05, 0.5, 0.05, dtype='d')),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart']), # CHANGE FOR IBES
    'max_bin': hp.choice('max_bin', [255]),
    'num_leaves': hp.choice('num_leaves', np.arange(50, 200, 30, dtype=int)),

    # avoid overfit
    'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(10, 310, 100, dtype=int)),
    'feature_fraction': hp.choice('feature_fraction', np.arange(0.7, 1, 0.1, dtype='d')),
    'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.7, 1, 0.1, dtype='d')),
    'bagging_freq': hp.choice('bagging_freq', [2, 4, 8]),
    'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0, 0.1, 0.02, dtype='d')),
    'lambda_l1': hp.choice('lambda_l1', np.arange(1, 20, 5, dtype=int)),
    # 'lambda_l2': hp.choice('lambda_l2', np.arange(350, 450, 20, dtype=int)),

    # parameters won't change
    # 'boosting_type': 'gbdt',  # past:  hp.choice('boosting_type', ['gbdt', 'dart']
    'objective': 'regression_l1',
    'verbose': 1,
    # 'metric': 'multi_error',
    'num_threads': 16  # for the best speed, set this to the number of real CPU cores
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

    gbm.save_model('mode.txt')

    '''Evaluation on Test Set'''
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
                'explained_variance_score': explained_variance_score(Y_test, Y_test_pred),
                'max_error': max_error(Y_test, Y_test_pred),
                # 'mse': mean_squared_error(Y_test, Y_test_pred),
                'median_absolute_error': median_absolute_error(Y_test, Y_test_pred),
                'r2_score': r2_score(Y_test, Y_test_pred),
                'status': STATUS_OK}

    sql_result.update(space)
    sql_result.update(result)
    sql_result['finish_timing'] = dt.datetime.now()

    pt = pd.DataFrame.from_records([sql_result], index=[0])

    print('sql_result_before writing: ', sql_result)

    # with engine.connect() as conn:
    #     pt.to_sql('results_lightgbm', con=conn, index=False, if_exists='append')
    # engine.dispose()

    return result['mae_valid']

def HPOT(space, max_evals):
    ''' use hyperopt on each set '''
    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    print(best)
    # return best

if __name__ == "__main__":

    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010, 201020, 502030, 401010]

    try:
        db_last = pd.read_sql("SELECT * FROM lightgbm_results order by finish_timing desc LIMIT 1", engine)  # identify current # trials from past execution
        db_last_trial = db_last['trial']
    except:
        db_last_trial = 0

    # parser
    resume = False
    qcut_q = 3
    sample_no = 1
    ''' # change to 28 for official run <- check '''

    sql_result = {}
    sql_result['name'] = 'trial'
    sql_result['trial'] = db_last_trial + 1
    sql_result['qcut_q'] = qcut_q

    # roll over each round
    period_1 = dt.datetime(2013, 3, 31)

    for i in tqdm(range(sample_no)):                # roll over testing period
        testing_period = period_1 + i * relativedelta(months=3)

        for icb_code in indi_models:                # roll over industries

            sample_set, cut_bins, cv = load_data(icb_code, testing_period, qcut_q).split_valid()
            sql_result['cut_bins'] = cut_bins['ni']

            for train_index, valid_index in cv:     # roll over 5 cross validation set
                X_train, X_valid = sample_set['train_x'][train_index], sample_set['train_x'][valid_index]
                Y_train, Y_valid = sample_set['train_ni'][train_index], sample_set['train_ni'][valid_index]  # lightGBM use Net Income as Y
                print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

                HPOT(space, max_evals=10)
                # exit(0)
