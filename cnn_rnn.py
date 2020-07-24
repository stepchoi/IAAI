import numpy as np
import os
import argparse
import pandas as pd
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import r2_score, mean_absolute_error

from tensorflow.python.keras import callbacks, optimizers, initializers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GRU, Dropout, Flatten,  LeakyReLU, Input, Concatenate, Reshape, Lambda, Conv2D

from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from load_data_rnn import load_data
from LightGBM import read_db_last
import matplotlib.pyplot as plt

import tensorflow as tf                             # avoid error in Tensorflow initialization
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--add_ind_code', type=int, default=0)
args = parser.parse_args()

space = {
    'learning_rate': hp.choice('lr', [1, 2, 3, 4, 5]), # drop 7
    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
    'kernel_size': hp.choice('kernel_size', [32, 128, 384]), #CNN kernel size - num of different "scenario"
    'num_gru_layer': hp.choice('num_gru_layer', [1, 2, 3]),     # number of layers # drop 1, 2
    'gru_nodes_mult': hp.choice('gru_nodes_mult', [0, 1]),      # nodes growth rate *1 or *2
    'gru_nodes': hp.choice('gru_nodes', [4, 8]),    # start with possible 4 nodes -- 8, 8, 16 combination possible
    'gru_dropout': hp.choice('gru_drop', [0.25, 0.5]),


    'activation': hp.choice('activation', ['tanh']),
    'batch_size': hp.choice('batch_size', [64, 128, 512]), # drop 1024
}

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def rnn_train(space): #functional
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''
    params = space.copy()
    print(params)

    lookback = 20                   # lookback = 5Y * 4Q = 20Q
    x_fields = X_train.shape[2]    # x_fields differ depending on whether include ibes ratios

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
        # extra.update(input_shape=(lookback, number_of_kernels * 2))
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
                      callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='auto')]  # add callbacks
    lr_val = 10 ** -int(params['learning_rate'])
    adam = optimizers.Adam(lr=lr_val)
    model.compile(adam, loss='mae')

    model.summary()

    history = model.fit(X_train, Y_train, epochs=200, batch_size=params['batch_size'],
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

    sql_result.update(space)
    sql_result.update(result)
    sql_result['finish_timing'] = dt.datetime.now()

    print('sql_result_before writing: ', sql_result)
    hpot['all_results'].append(sql_result.copy())

    # with engine.connect() as conn:
    #     pd.DataFrame.from_records(sql_result, index=[0]).to_sql('results_cnn_rnn', con=conn, index=False,
    #                                                             if_exists='append', method='multi')
    # engine.dispose()
    #
    # plot_history(history, sql_result['trial_lgbm'], sql_result['mae_test'])  # plot training history

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)
        hpot['best_history'] = history
        hpot['best_trial'] = sql_result['trial_lgbm']

    sql_result['trial_lgbm'] += 1

    return result['mae_valid']

def HPOT(space, max_evals = 10):
    ''' use hyperopt on each set '''

    hpot['best_mae'] = 1  # record best training (min mae_valid) in each hyperopt
    hpot['all_results'] = []

    trials = Trials()
    best = fmin(fn=eval, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    print(hpot['best_stock_df'])

    with engine.connect() as conn:
        pd.DataFrame(hpot['all_results']).to_sql('results_cnn_rnn', con=conn, index=False, if_exists='append', method='multi')
        hpot['best_stock_df'].to_sql('results_cnn_rnn_stock', con=conn, index=False, if_exists='append', method='multi')
    engine.dispose()

    # plot_history(hpot['best_history'], hpot['best_trial'], hpot['best_mae'])  # plot training history

    sql_result['trial_hpot'] += 1

    return best

def plot_history(history, trial, mae):
    ''' plot the training loss history '''

    history_dict = history.history
    epochs = range(10, len(history_dict['loss'])+1)

    plt.plot(epochs, history_dict['loss'][9:], 'bo', label='training loss')
    plt.plot(epochs, history_dict['val_loss'][9:], 'b', label='validation loss')
    plt.title('dense - training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('results_rnn/plot_cnn_dnn_{} {}.png'.format(trial, round(mae,4)))
    plt.close()

def pred_to_sql(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    # print('stock-wise prediction: ', df)

    return df

if __name__ == "__main__":

    sql_result = {}
    hpot = {}

    # default params for load_data
    period_1 = dt.datetime(2013,3,31)
    sample_no = 25
    load_data_params = {'exclude_fwd': True,
                        'y_type': 'ibes',
                        'qcut_q': 10}

    # these are parameters used to load_data
    sql_result['qcut_q'] = load_data_params['qcut_q']
    sql_result['name'] = 'without ibes -2'
    db_last_param, sql_result = read_db_last(sql_result, 'results_cnn_rnn')

    data = load_data(macro_monthly=True)

    add_ind_code = args.add_ind_code # 1 means add industry code as X; 2 mesns add sector code as X
    data.split_entire(add_ind_code=add_ind_code)
    sql_result['icb_code'] = add_ind_code

    for i in tqdm(range(1)):  # roll over testing period
        testing_period = period_1 + i * relativedelta(months=3)
        sql_result['testing_period'] = testing_period

        train_x, train_y, X_test, Y_test, cv, test_id, x_col = data.split_train_test(testing_period, **load_data_params)
        print(x_col)
        X_test = np.expand_dims(X_test, axis=3)

        cv_number = 1
        for train_index, test_index in cv:
            sql_result['cv_number'] = cv_number

            X_train = np.expand_dims(train_x[train_index], axis=3)
            Y_train = train_y[train_index]
            X_valid = np.expand_dims(train_x[test_index], axis=3)
            Y_valid = train_y[test_index]

            print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

            HPOT(space, 10)
            cv_number += 1


