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

from load_data_lgbm import load_data

space = {

    # 'num_GRU_layer': hp.choice('num_GRU_layer', [1, 2, 3]),
    'num_Dense_layer': hp.choice('num_Dense_layer', [1, 2, 3]),    # number of layers

    'neurons_layer_1': hp.choice('neurons_layer_1', [4, 8, 16]),
    'neurons_layer_2': hp.choice('neurons_layer_2', [4, 8, 16]),
    'neurons_layer_3': hp.choice('neurons_layer_3', [4, 8, 16]),

    'batch_size': hp.choice('batch_size', [512, 1024, 2048]),
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

    model.fit(X_train, Y_train, epochs=20, batch_size=params['batch_size'], validation_data=(X_valid, Y_valid), verbose=1)
    model.summary()


    train_mae = model.evaluate(X_train, Y_train,  verbose=1)
    valid_mae = model.evaluate(X_valid, Y_valid, verbose=1)
    test_mae = model.evaluate(X_test, Y_test, verbose=1)

    print(train_mae, valid_mae, test_mae)

    return train_mae, valid_mae, test_mae

def f(space):

    train_mae, valid_mae, test_mae = dense_train(space)

    result = {'mae_train': train_mae,
              'mae_valid': valid_mae,
              'mae_test': test_mae,
              'status': STATUS_OK}

    print(space)
    print(result)
    sql_result.update(result)
    sql_result.update(space)

    with engine.connect() as conn:
        pd.DataFrame(sql_result).to_sql('results_dense', con=conn, if_exists='append')
    engine.dispose()

    return result

def HPOT(space):


    trials = Trials()
    best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    return best

# def history_log():
#     num_epoch = 200
#     all_mae_histories = []
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)



# try functional API?


if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    sql_result = {}

    # these are parameters used to load_data
    icb_code = 301010
    exclude_fwd = False
    use_median = True
    chron_valid = False
    testing_period = dt.datetime(2013,3,31)
    qcut_q = 10

    data = load_data()
    data.split_icb(icb_code)
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

        dense_train(space)