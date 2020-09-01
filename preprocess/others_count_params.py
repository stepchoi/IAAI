import numpy as np
import os
import gc
import argparse
import pandas as pd
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import r2_score, mean_absolute_error

from tensorflow.python.keras import callbacks, optimizers, initializers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GRU, Dropout, Flatten,  LeakyReLU, Input, Concatenate, Reshape, Lambda, Conv2D
from tensorflow.python.keras import backend as K

from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from load_data_rnn import load_data
from lgbm import read_db_last
import matplotlib.pyplot as plt
from hyperspace_lgbm import find_hyperspace

import tensorflow as tf                             # avoid error in Tensorflow initialization
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

params = ['batch_size', 'dropout', 'init_nodes', 'learning_rate', 'mult_freq', 'mult_start', 'nodes_mult', 'num_gru_layer',
          'num_Dense_layer', 'num_nodes', 'gru_dropout', 'gru_nodes', 'gru_nodes_mult', 'icb_code', 'kernel_size']

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'

def download(tname, r_name):
    ''' donwload results from results_lightgbm '''

    query = "select * from (select DISTINCT *, min(mae_valid) over (partition by trial_hpot, exclude_fwd, icb_code) " \
            "as min_thing from results_{})t where mae_valid = min_thing and name = '{}' ".format(tname, r_name)

    with engine.connect() as conn:
        results = pd.read_sql(query, con=conn)
    engine.dispose()

    print(results.columns)

    results = results.drop_duplicates(subset=['icb_code', 'identifier', 'testing_period', 'cv_number','y_type'], keep='last')

    return results.filter(params)

def cnn_rnn(space, x_fields): #functional
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''
    params = space.copy()
    print(params)

    lookback = 20                   # lookback = 5Y * 4Q = 20Q

    #FUNCTIONAL  - refer to the input after equation formuala with (<prev layer>)
    #pseudo-code---------------------------------------------------------------------------------------------------------

    kernel_size =params['kernel_size'] # of different "scenario"
    num_nodes = params['gru_nodes']

    #CNN - use one conv layer to encode the 2D vector into 2D lookback X 1 vector
    input_img = Input(shape=(lookback, x_fields, 1))

    #reduce the 2D vector in lookback X 1 where the ONE number indicated one of num_kern financial "scenarios"
    c_1 = Conv2D(kernel_size, (1, x_fields), strides=(1, x_fields), padding='valid', name='conv1')(input_img)
    c_1 = LeakyReLU(alpha=0.1)(c_1)

    g_1 = Reshape((lookback, kernel_size))(c_1) # reshape for GRU

    #GRU part ---------------------------------
    for i in range(params['num_gru_layer']):
        extra = dict(return_sequences=True) # need to iterative
        temp_nodes = int(max(params['gru_nodes'] * (2 ** (params['gru_nodes_mult'] * i)), 8)) # nodes grow at 2X or stay same - at least 8 nodes

        if i == params['num_gru_layer'] - 1:
            extra = dict(return_sequences=False)  # last layer does not output the whole sequence
            g_1_2 = GRU(temp_nodes, **extra)(g_1) # this is the forecast state
            extra = dict(return_sequences=True)
            g_1 = GRU(1, dropout=0, **extra)(g_1)
        elif i == 0:
            g_1 = GRU(temp_nodes, **extra)(g_1)
        else:
            g_1 = GRU(temp_nodes, dropout=params['gru_dropout'], **extra)(g_1)

    g_1 = Flatten()(g_1)    # convert 3d sequence(?,?,1) -> 2d (?,?)

    #join the return sequence and forecast state
    f_x = Concatenate(axis=1)([g_1, g_1_2])
    f_x = Dense(lookback + 1)(f_x) #nodes = len return sequence +  1 for the forecast state
    f_x = Dense(1)(f_x)

    model = Model(input_img, f_x)
    # end of pseudo-code--------------------------------------------------------------------------------------------------

    callbacks_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
                      callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')]  # add callbacks
    lr_val = 10 ** -int(params['learning_rate'])
    adam = optimizers.Adam(lr=lr_val)
    model.compile(adam, loss='mae')

    return model

def rnn_top(space): #functional
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''
    params = space.copy()
    print(params)

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

    print(layers)

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

    return model

def count_p(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))

if __name__ == '__main__':
    r_name = 'new_without_ibes'
    # r_name = 'small_training_False_0'
    # r_name = 'top15'
    # r_name = 'industry_exclude'
    tname = 'cnn_rnn'

    # r_name = 'top15_lgbm'
    # tname = 'rnn_top'

    download(tname, r_name)