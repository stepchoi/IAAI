import numpy as np
import os
import argparse
import pandas as pd
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from keras import models, callbacks, optimizers
from keras.models import Model
from keras.layers import Dense, GRU, Dropout, Flatten,  LeakyReLU, Input, Concatenate, Reshape
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from load_data_rnn import load_data
from LightGBM import read_db_last
import matplotlib.pyplot as plt

space = {
    'learning_rate': hp.choice('lr', [2, 3, 4, 5, 7]),
    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
    'num_Dense_layer': hp.choice('num_Dense_layer', [1, 2, 3, ]),  # number of dense layers BEFORE GRU
    'num_nodes': hp.choice('num_nodes', [16, 32, 64 ]),  #nodes per layer BEFORE GRU

    'num_gru_layer': hp.choice('num_gru_layer', [1, 2, 3]),    # number of layers
    'gru_nodes_mult': hp.choice('gru_nodes_mult', [0, 1]),
    'gru_nodes': hp.choice('gru_nodes', [4, 8]), # start with possible 4 nodes -- 8, 8, 16 combination possible

    'num_gru_layer2': hp.choice('num_gru_layer', [1, 2, 3]),  # number of layers
    'gru_nodes_mult2': hp.choice('gru_nodes_mult', [0, 1]),
    'gru_nodes2': hp.choice('gru_nodes', [2, 4]),  # start with possible 4 nodes -- 8, 8, 16 combination possible

    'gru_dropout': hp.choice('gru_drop', [0.25, 0.5]),

    'activation': hp.choice('activation', ['tanh']),
    'batch_size': hp.choice('batch_size', [64, 128, 512, 1024]),
}

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def rnn_train(space): #functional
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''
    #lookback = quarters = 20?
    params = space.copy()
    print(params)

    lookback = 20                   # lookback = 5Y * 4Q = 20Q
    x_fields = X_train.shape[-1]    # x_fields differ depending on whether include ibes ratios

    #FUNCTIONAL  - refer to the input after equation formuala with (<prev layer>)
    #pseudo-code---------------------------------------------------------------------------------------------------------
    input_shape1 = (lookback, x_fields)
    input_img = Input(shape=input_shape1)           # input date: (batch_size, lookback, x_fields)
    input_img_flat = Flatten()(input_img)           # flatten layer for dense training

    input_shape2 = (lookback, 1)                    #JUST EARNINGS %
    input_img2 = Input(shape=input_shape2)          #JUST EARNINGS %

    num_layers =params['num_Dense_layer']
    num_nodes =params['num_nodes']

    # dense layers to start -------------------------------
    d_1 = Dense(num_nodes*lookback)(input_img) #first dense layer

    for i in range(num_layers - 1): #for second or third dense layers
        d_1 = Dense(num_nodes*lookback)(d_1)

    g_1 = Reshape((lookback, num_nodes))(d_1) # reshape for GRU

    #GRU part ---------------------------------
    for i in range(params['num_gru_layer']):
        extra = dict(return_sequences=True) # need to iterative
        temp_nodes = int(max(params['gru_nodes'] * (2 ** (params['num_gru_layer'] * i)), 8)) # nodes grow at 2X or stay same - at least 8 nodes

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
            # g_1 = Flatten()(g_1)


    #second GRU
    for i in range(params['num_gru_layer2']):
        extra = dict(return_sequences=True)  # need to iterative
        temp_nodes2 = int(min(params['gru_nodes2'] * (2 ** (params['num_gru_layer']) * (i))),4)  # nodes grow at 2X or stay same - at least 8 nodes
        if i == 0:
            # extra.update(input_shape=(lookback, number_of_kernels * 2))
            g_2 = GRU(temp_nodes2, **extra)(input_img2)
        elif i == params['num_gru_layer'] - 1:
            extra = dict(return_sequences=False)  # last layer does not output the whole sequence
            g_2_2 = GRU(temp_nodes, **extra)(g_2) # this is the forecast state
            extra = dict(return_sequences=True)
            g_2 = GRU(1, dropout=0, **extra)(g_2)
        else:
            g_2 = GRU(temp_nodes, dropout=gru_drop, **extra)(g_2)
            g_2 = Flatten()(g_2)

        # join the return sequence and forecast state
        f_x = Concatenate(axis=1)([g_1, g_1_2, g_2, g_2_2])
        f_x = Dense(lookback + 1)(f_x)  # nodes = len return sequence +  1 for the forecast state
        f_x = Flatten()(f_x)
        f_x = Dense(1)(f_x)


    model = Model([input_img, input_img2], f_x) # input BOTH images as an ARRAY
    # end of pseudo-code--------------------------------------------------------------------------------------------------

    callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='auto')
    lr_val = 10 ** -int(params['learning_rate'])
    adam = optimizers.Adam(lr=lr_val)
    model.compile(adam, loss='mae')

    history = model.fit(X_train, Y_train, epochs=200, batch_size=params['batch_size'], validation_data=(X_valid, Y_valid), verbose=1)
    model.summary()

    train_mae = model.evaluate(X_train, Y_train,  verbose=1)
    valid_mae = model.evaluate(X_valid, Y_valid, verbose=1)
    test_mae = model.evaluate(X_test, Y_test, verbose=1)
    Y_test_pred = model.predict(X_test)

    return train_mae, valid_mae, test_mae, Y_test_pred, history

def eval(space):
    ''' train & evaluate each of the rnn model '''

    train_mae, valid_mae, test_mae, Y_test_pred, history = rnn_train(space)

    result = {'mae_train': train_mae,
              'mae_valid': valid_mae,
              'mae_test': test_mae,
              'status': STATUS_OK}

    print(space)
    print(result)
    sql_result.update(space)
    sql_result.update(result)
    sql_result['finish_timing'] = dt.datetime.now()

    with engine.connect() as conn:  # save training results
        pd.DataFrame(sql_result, index=[0]).to_sql('results_rnn', con=conn, index=False, if_exists='append')
    engine.dispose()

    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)

    plot_history(history)  # plot history (epoch -> training / validation loss) find optimal epoch needed

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
        hpot['best_stock_df'].to_sql('results_rnn_stock', con=conn, index=False, if_exists='append')
    engine.dispose()

    sql_result['trial_hpot'] += 1

    return best

def plot_history(history):
    ''' plot the training loss history '''

    history_dict = history.history
    epochs = range(10, len(history_dict['loss'])+1)

    plt.plot(epochs, history_dict['loss'][9:], 'bo', label='training loss')
    plt.plot(epochs, history_dict['val_loss'][9:], 'b', label='validation loss')
    plt.title('dense - training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    plt.savefig('results_dense/plot_rnn_{}.png'.format(sql_result['trial_lgbm']))
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
    exclude_fwd = False
    use_median = True
    chron_valid = False
    sql_result['name'] = 'new entire'

    # these are parameters used to load_data
    period_1 = dt.datetime(2018,3,31)
    sql_result['qcut_q'] = 10
    sample_no = 1
    db_last_param, sql_result = read_db_last(sql_result, 'results_rnn3', first=True)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    data = load_data()

    for add_ind_code in [0]: # 1 means add industry code as X
        data.split_entire(add_ind_code=add_ind_code)
        sql_result['icb_code'] = add_ind_code

        for i in tqdm(range(1)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            train_x, train_y, X_test, Y_test, cv, test_id, x_col = data.split_train_test(sql_result['testing_period'],
                                                                                         sql_result['qcut_q'],
                                                                                         y_type='ibes')

            sql_result['x_fields'] = str(x_col)     # record x_fields used to DB TABLE

            eps_index = x_col.index('eps1tr12')     # find index for earnings column
            train_eps = train_x[:, :, eps_index]    # slice earnings column for training
            eps_test = X_test[:, :, eps_index]

            cv_number = 1
            for train_index, test_index in cv:      # split train / valid
                X_train = train_x[train_index]
                Y_train = train_y[train_index]
                X_valid = train_x[test_index]
                Y_valid = train_y[test_index]

                eps_train = train_eps[train_index]  # split train / valid for eps column array
                eps_valid = train_eps[train_index]

                print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

                HPOT(space, 10)


