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

from load_data_rnn import load_data

def RNN_train():

    model = models.Sequential()
    model.add(Dense(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mae')

    ## try LSTM / GRU models?
    ## Callback?

    ## add regularization?



    model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_valid, Y_valid), verbose=1)


    loss_train, train_mae = model.evaluate(X_train, Y_train, batch_size=128, verbose=1)
    loss_valid, valid_mae = model.evaluate(X_valid, Y_valid, batch_size=128, verbose=1)
    loss_valid, test_mae = model.evaluate(X_test, Y_test, batch_size=128, verbose=1)

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

    icb_code = 301010
    testing_period = dt.datetime(2013, 3, 31)
    qcut_q = 10

    data = load_data()
    data.split_icb(icb_code)
    train_x, train_y, X_test, Y_test, cv = data.split_train_test(testing_period, qcut_q, y_type='ni')

    for train_index, test_index in cv:
        X_train = train_x[train_index]
        Y_train = train_y[train_index]
        X_valid = train_x[test_index]
        Y_valid = train_y[test_index]

        print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

        RNN_train()