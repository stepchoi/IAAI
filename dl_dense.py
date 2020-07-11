import numpy as np
import os
import argparse
import pandas as pd
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from keras import models, callbacks
from keras.layers import Dense, GRU, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from load_data_lgbm import load_data
from LightGBM import pred_to_sql, read_db_last

space = {

    # 'num_GRU_layer': hp.choice('num_GRU_layer', [1, 2, 3]),
    'num_Dense_layer': hp.choice('num_Dense_layer', [1, 2, 3]),    # number of layers

    'neurons_layer_1': hp.choice('neurons_layer_1', [8, 16, 32, 64]),
    'neurons_layer_2': hp.choice('neurons_layer_2', [8, 16, 32, 64]),
    'neurons_layer_3': hp.choice('neurons_layer_3', [8, 16, 32, 64]),

    'batch_size': hp.choice('batch_size', [64, 128, 512]),
    # 'dropout': hp.choice('dropout', [0, 0.2, 0.4])

}

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def dense_train(space):

    params = space.copy()
    print(params)

    model = models.Sequential()
    for i in range(params['num_Dense_layer']):
        model.add(Dense(params['neurons_layer_{}'.format(i+1)], activation='tanh'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mae')

    ## try LSTM / GRU models?
    ## Callback?

    ## add regularization?

    model.fit(X_train, Y_train, epochs=100, batch_size=params['batch_size'], validation_data=(X_valid, Y_valid), verbose=1)
    model.summary()


    train_mae = model.evaluate(X_train, Y_train,  verbose=1)
    valid_mae = model.evaluate(X_valid, Y_valid, verbose=1)
    test_mae = model.evaluate(X_test, Y_test, verbose=1)
    Y_test_pred = model.predict(X_test)

    return train_mae, valid_mae, test_mae, Y_test_pred

def eval(space):

    train_mae, valid_mae, test_mae, Y_test_pred = dense_train(space)

    result = {'mae_train': train_mae,
              'mae_valid': valid_mae,
              'mae_test': test_mae,
              'status': STATUS_OK}

    print(space)
    print(result)
    sql_result.update(space)
    sql_result.update(result)

    hpot['all_results'].append(sql_result.copy())
    sql_result['finish_timing'] = dt.datetime.now()

    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] > hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)

    sql_result['trial_rnn'] += 1

    return result['mae_valid']

def HPOT(space):

    hpot['best_mae'] = 1  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

    with engine.connect() as conn:
        pd.DataFrame(hpot['all_results']).to_sql('results_dense_new', con=conn, index=False, if_exists='append')
        hpot['best_stock_df'].to_sql('results_dense_stock', con=conn, index=False, if_exists='append')
    engine.dispose()

    sql_result['trial_hpot'] += 1

    return best

# def history_log():
#     num_epoch = 200
#     all_mae_histories = []
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)



# try functional API?


if __name__ == "__main__":

    sql_result = {}
    hpot = {}
    exclude_fwd = False
    use_median = True
    chron_valid = False
    sql_result['name'] = 'sector'

    # these are parameters used to load_data
    indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                   201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    period_1 = dt.datetime(2013,3,31)
    qcut_q = 10
    sample_no = 25
    db_last_param, sql_result = read_db_last(sql_result)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    data = load_data()

    for icb_code in indi_models:  # roll over industries (first 2 icb code)
        data.split_icb(icb_code)
        sql_result['icb_code'] = icb_code

        for i in tqdm(range(sample_no)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, qcut_q,
                                                                              y_type='ni',
                                                                              exclude_fwd=exclude_fwd,
                                                                              use_median=use_median,
                                                                              chron_valid=chron_valid)

            X_test =  np.nan_to_num(sample_set['test_x'], nan=0)
            Y_test = sample_set['test_y']

            for train_index, test_index in cv:
                X_train = np.nan_to_num(sample_set['train_x'][train_index], nan=0)
                Y_train = sample_set['train_y'][train_index]
                X_valid =  np.nan_to_num(sample_set['train_x'][test_index], nan=0)
                Y_valid = sample_set['train_y'][test_index]

                print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

                try:
                    HPOT(space)
                except:
                    continue


