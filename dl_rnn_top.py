import numpy as np
import os
import gc
import argparse
import pandas as pd
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import r2_score, mean_absolute_error
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from tensorflow.python.keras import callbacks, optimizers, initializers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GRU, Dropout, Flatten,  LeakyReLU, Input, Concatenate, Reshape, Lambda, Conv2D
from tensorflow.python.keras import backend as K

from load_data_rnn import load_data

import tensorflow as tf                             # avoid error in Tensorflow initialization
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


space = {
    'learning_rate': hp.choice('lr', [1, 2]), # => 1e-x - learning rate - REDUCE space later - correlated to batch size

    'num_gru_layer': hp.choice('num_gru_layer', [2, 3, 4]),     # number of layers # drop 1
    'gru_nodes_mult': hp.choice('gru_nodes_mult', [0, 1]),      # nodes growth rate *1 or *2
    'gru_nodes': hp.choice('gru_nodes', [1, 2]),    # start with possible 4 nodes -- 8, 8, 16 combination possible

    # 'gru_dropout': 0,
    'gru_dropout': hp.choice('gru_drop', [0.1, 0.25]),

    'activation': 'tanh',
    'batch_size': 128 # drop 64, 512, 1024
}

def gpu_mac_address(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def rnn_train(space): #functional
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''
    params = space.copy()

    lookback = 20                   # lookback = 5Y * 4Q = 20Q
    x_fields = 10                   # lgbm top15 features -> 10 features in dense

    input_img = Input(shape=(lookback, x_fields))
    layers = []

    for col in range(10):   # build model for each feature

        g_1 = K.expand_dims(input_img[:,:,col], axis=2)    # slide input img to certain feature: shape = (samples, 20, 1)

        for i in range(1, params['num_gru_layer']):
            temp_nodes = int(min(params['gru_nodes'] * (2 ** (params['gru_nodes_mult'] * (i-1))), 8))  # nodes grow at 2X or stay same - at least 8 nodes

            if i == params['num_gru_layer'] - 1:
                g_1_2 = GRU(temp_nodes, return_sequences=False)(g_1)  # this is the forecast state, last layer does not output the whole sequence
                g_1 = GRU(1, dropout=0, return_sequences=True)(g_1)
            elif i == 1:
                g_1 = GRU(temp_nodes, return_sequences=True)(g_1)
            else:
                g_1 = GRU(temp_nodes, dropout=params['gru_dropout'], return_sequences=True)(g_1)

        g_1 = Flatten()(g_1)  # convert 3d sequence(?,?,1) -> 2d (?,?)
        layers.extend([g_1, g_1_2])

    # join the return sequence and forecast state
    f_x = Concatenate(axis=1)(layers)
    f_x = Dense(lookback + 1)(f_x)  # nodes = len return sequence +  1 for the forecast state
    f_x = Dense(1)(f_x)

    model = Model(input_img, f_x)
    # end of pseudo-code--------------------------------------------------------------------------------------------------

    callbacks_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
                      callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')]  # add callbacks
    lr_val = 10 ** -int(params['learning_rate'])
    adam = optimizers.Adam(lr=lr_val)
    model.compile(adam, loss='mae')

    model.summary()

    history = model.fit(X_train, Y_train, epochs=50, batch_size=params['batch_size'],
                        validation_data=(X_valid, Y_valid), verbose=1, callbacks=callbacks_list)

    Y_test_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)
    Y_valid_pred = model.predict(X_valid)

    return Y_test_pred, Y_train_pred, Y_valid_pred, history

def eval(space):
    ''' train & evaluate each of the dense model '''

    Y_test_pred, Y_train_pred, Y_valid_pred, history = rnn_train(space)

    result = {'mae_train': mean_absolute_error(Y_train, Y_train_pred),
              'mae_valid': mean_absolute_error(Y_valid, Y_valid_pred),
              'mae_test': mean_absolute_error(Y_test, Y_test_pred),
              'r2_train': r2_score(Y_train, Y_train_pred),
              'r2_valid': r2_score(Y_valid, Y_valid_pred),
              'r2_test': r2_score(Y_test, Y_test_pred),
              'status': STATUS_OK}

    K.clear_session()
    return result['mae_valid']

def HPOT(space, max_evals = 10):
    ''' use hyperopt on each set '''

    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    return best

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_number', type=int, default=1)
    args = parser.parse_args()
    gpu_mac_address(args)           # GPU set up

    # default params for load_data
    period_1 = dt.datetime(2013,4,1)
    sample_no = 21
    load_data_params = {'qcut_q': 10, 'y_type': 'ibes', 'exclude_fwd': False, 'eps_only': False, 'top15': 'lgbm'}

    data = load_data(macro_monthly=True)        # load data step 1
    data.split_entire()    # load data step 2

    for i in tqdm(range(21)):  # roll over testing period
        testing_period = period_1 + i * relativedelta(months=3) - relativedelta(days=1)
        train_x, train_y, X_test, Y_test, cv, test_id, x_col, cut_bins = data.split_train_test(testing_period, **load_data_params)      # load data step 3

        for train_index, test_index in cv:      # roll over 5-fold cross validation
            X_train = train_x[train_index]
            Y_train = train_y[train_index]
            X_valid = train_x[test_index]
            Y_valid = train_y[test_index]

            HPOT(space, 10)
            gc.collect()


