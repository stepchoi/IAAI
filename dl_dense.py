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

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

def dense_train():

    model = models.Sequential()
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mae')

    ## try LSTM / GRU models?
    ## Callback?

    ## add regularization?



    model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_valid, Y_valid), verbose=1)
    model.summary()


    train_mae = model.evaluate(X_train, Y_train,  verbose=1)
    valid_mae = model.evaluate(X_valid, Y_valid, verbose=1)
    test_mae = model.evaluate(X_test, Y_test, verbose=1)

    print(train_mae, valid_mae, test_mae)

    return train_mae, valid_mae, test_mae

# def history_log():
#     num_epoch = 200
#     all_mae_histories = []
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_histories.append(mae_history)



# try functional API?


if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

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

        dense_train()