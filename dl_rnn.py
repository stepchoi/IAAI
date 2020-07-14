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

from load_data_rnn import load_data
from LightGBM import read_db_last
import matplotlib.pyplot as plt

space = {

    'num_gru_layer': hp.choice('num_gru_layer', [1, 2, 3]),    # number of layers
    # 'num_dense_layer': hp.choice('num_Dense_layer', [0]),  # number of layers

    'gru_1': hp.choice('gru_1', [8, 16, 32, 64]),
    'dropout_1': hp.choice('dropout_1', [0, 0.2, 0.4]),
    'recurrent_dropout_1': hp.choice('recurrent_dropout_1', [0, 0.2, 0.4]),

    'gru_2': hp.choice('gru_2', [8, 16, 32, 64]),
    'dropout_2': hp.choice('dropout_2', [0, 0.2, 0.4]),
    'recurrent_dropout_2': hp.choice('recurrent_dropout_2', [0, 0.2, 0.4]),

    'gru_3': hp.choice('gru_3', [16, 32, 64, 128]),
    'dropout_3': hp.choice('dropout_3', [0, 0.2, 0.4]),
    'recurrent_dropout_3': hp.choice('recurrent_dropout_3', [0, 0.2, 0.4]),

    'activation': hp.choice('activation', ['relu','tanh']),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 512]),

}

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def dense_train(space):

    params = space.copy()
    print(params)

    model = models.Sequential()
    for i in range(params['num_gru_layer']):
        model.add(GRU(params['gru_{}'.format(i+1)], activation=params['activation'],
                      dropout=params['dropout_{}'.format(i+1)], recurrent_dropout=params['recurrent_dropout_{}'.format(i+1)],
                      return_sequences=not(i==params['num_gru_layer']-1), input_shape=(X_train.shape[1], X_train.shape[2]), ))

    # model.add(Flatten())
    #
    # if params['num_dense_layer'] != 0:
    #     for i in range(params['num_dense_layer']):
    #         model.add(Dense(params['neurons_layer_{}'.format(i + 1)], activation=params['activation']))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mae')

    history = model.fit(X_train, Y_train, epochs=100, batch_size=params['batch_size'], validation_data=(X_valid, Y_valid), verbose=1)
    model.summary()

    train_mae = model.evaluate(X_train, Y_train,  verbose=1)
    valid_mae = model.evaluate(X_valid, Y_valid, verbose=1)
    test_mae = model.evaluate(X_test, Y_test, verbose=1)
    Y_test_pred = model.predict(X_test)

    return train_mae, valid_mae, test_mae, Y_test_pred, history

def eval(space):

    train_mae, valid_mae, test_mae, Y_test_pred, history = dense_train(space)

    result = {'mae_train': train_mae,
              'mae_valid': valid_mae,
              'mae_test': test_mae,
              'status': STATUS_OK}

    print(space)
    print(result)
    sql_result.update(space)
    sql_result.update(result)
    sql_result['finish_timing'] = dt.datetime.now()

    with engine.connect() as conn:
        pd.DataFrame(sql_result, index=[0]).to_sql('results_rnn', con=conn, index=False, if_exists='append')
    engine.dispose()

    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)

    plot_history(history)  # plot history

    sql_result['trial_lgbm'] += 1

    return result['mae_valid']

def HPOT(space, max_evals = 10):

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
    db_last_param, sql_result = read_db_last(sql_result, 'results_dense')  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    data = load_data()

    for add_ind_code in [0]: # 1 means add industry code as X
        data.split_entire(add_ind_code=add_ind_code)
        sql_result['icb_code'] = add_ind_code

        for i in tqdm(range(1)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            train_x, train_y, X_test, Y_test, cv, test_id = data.split_train_test(testing_period, sql_result['qcut_q'], y_type='ni')

            cv_number = 1
            for train_index, test_index in cv:
                X_train = train_x[train_index]
                Y_train = train_y[train_index]
                X_valid = train_x[test_index]
                Y_valid = train_y[test_index]

                print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

                HPOT(space, 10)
                exit(0)


