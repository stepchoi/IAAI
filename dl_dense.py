import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import ast
import argparse
from tqdm import tqdm
import os

from hyperopt import fmin, tpe, STATUS_OK, Trials
from tensorflow.python.keras import callbacks, optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras import backend as K
from sklearn.metrics import r2_score, mean_absolute_error

from load_data_lgbm import load_data
from hyperspace_dense import find_hyperspace
import matplotlib.pyplot as plt

import tensorflow as tf                             # avoid error in Tensorflow initialization
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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

def dense_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()

    input_shape = (X_train.shape[-1],)      # input shape depends on x_fields used
    input_img = Input(shape=input_shape)


    init_nodes = params['init_nodes']
    nodes_mult = params['nodes_mult']
    mult_freq = params['mult_freq']
    mult_start = params['mult_start']
    num_Dense_layer = params['num_Dense_layer']

    if num_Dense_layer < 4:
        params['init_nodes'] = init_nodes = 16

    d_1 = Dense(init_nodes, activation=params['activation'])(input_img)  # remove kernel_regularizer=regularizers.l1(params['l1'])
    d_1 = Dropout(params['dropout'])(d_1)

    nodes = [init_nodes]
    for i in range(1, num_Dense_layer):
        temp_nodes = int(min(init_nodes * (2 ** (nodes_mult * max((i - mult_start + 3)//mult_freq, 0))), params['end_nodes'])) # nodes grow at 2X or stay same - at most 128 nodes
        d_1 = Dense(temp_nodes, activation=params['activation'])(d_1)  # remove kernel_regularizer=regularizers.l1(params['l1'])
        nodes.append(temp_nodes)

        if i != num_Dense_layer - 1:    # last dense layer has no dropout
            d_1 = Dropout(params['dropout'])(d_1)

    f_x = Dense(1)(d_1)

    callbacks_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
                      callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')]  # add callbacks
    lr_val = 10 ** -int(params['learning_rate'])

    adam = optimizers.Adam(lr=lr_val)
    model = Model(input_img, f_x)
    model.compile(adam, loss='mae')
    model.summary()

    history = model.fit(X_train, Y_train, epochs=50, batch_size=params['batch_size'], validation_data=(X_valid, Y_valid),
                        callbacks=callbacks_list, verbose=1)

    Y_test_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)
    Y_valid_pred = model.predict(X_valid)

    return Y_test_pred, Y_train_pred, Y_valid_pred, history

def eval(space):
    ''' train & evaluate each of the dense model '''

    Y_test_pred, Y_train_pred, Y_valid_pred, history = dense_train(space)       # train model and evaluate

    result = {'mae_train': mean_absolute_error(Y_train, Y_train_pred),      # evaluate metrices
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

    parser = argparse.ArgumentParser()  # setting for different configuration
    parser.add_argument('--objective', default='regression_l1')  # regression_l2 = optimizing mse
    parser.add_argument('--exclude_fwd', default=False, action='store_true')  # True = without ibes as X
    parser.add_argument('--sample_type', default='industry')  # sampling type
    parser.add_argument('--y_type', default='ibes')  # ibes_qoq for qoq type growth prediction
    parser.add_argument('--qcut_q', default=10, type=int)
    parser.add_argument('--sample_ratio', default=1, type=float)  # 0.5 = select 50% random sample from original data set
    parser.add_argument('--nthread', default=12, type=int)
    args = parser.parse_args()

    gpu_mac_address(args)

    # training / testing sets split par
    if args.sample_type == 'industry':
        partitions = [11, 20, 30, 35, 40, 45, 51, 60, 65]  # 11 represents industry (10 + 15); 51 represents (50 + 55)
    elif args.sample_type == 'entire':
        partitions = [0]  # 0 represents aggregate model

    period_1 = dt.datetime(2013, 3, 31)     # starting point for first testing set
    base_space = {'verbose': -1,
                  'num_threads': args.nthread}  # for the best speed, set this to the number of real CPU cores

    # create dict storing values/df used in training
    sample_no = args.sample_no  # number of training/testing period go over ( 25 = until 2019-3-31)

    load_data_params = {'exclude_fwd': args.exclude_fwd,
                        'y_type': args.y_type,
                        'qcut_q': args.qcut_q,
                        }

    # load data from raw data files in DB
    data = load_data(sample_ratio=args.sample_ratio)          # load data step 1

    for icb_code in partitions:   # roll over industries (first 2 icb code)
        data.split_industry(icb_code)      # load data step 2

        for i in tqdm(range(sample_no)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, **load_data_params)     # load data step 3 - organize final x/y arrays

            X_test = np.nan_to_num(sample_set['test_x'], nan=0)     # fill NaN for X
            Y_test = sample_set['test_y']

            for train_index, test_index in cv:      # cross validation
                X_train = np.nan_to_num(sample_set['train_x'][train_index], nan=0)
                Y_train = sample_set['train_y'][train_index]
                X_valid = np.nan_to_num(sample_set['train_x'][test_index], nan=0)
                Y_valid = sample_set['train_y'][test_index]

                space = find_hyperspace(args)
                HPOT(space, 10)



