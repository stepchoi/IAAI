import numpy as np
import os
import argparse
import pandas as pd
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from keras import models, callbacks, optimizers
from keras.layers import Dense, GRU, Dropout, Flatten, LeakyReLU
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from load_data_lgbm import load_data
from LightGBM import read_db_last
import matplotlib.pyplot as plt

space = {

    # 'num_GRU_layer': hp.choice('num_GRU_layer', [1, 2, 3]),
    'num_Dense_layer': hp.choice('num_Dense_layer', [2, 3, 4, 5]),  # number of layers ONE layer is TRIVIAL
    'learning_rate': hp.choice('lr', [2, 3, 4]),    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
                                                    # remove lr = 5 & 7 after tuning

    'neurons_layer_1': hp.choice('neurons_layer_1', [16, 32]),
    'dropout_1': hp.choice('dropout_1', [0.25, 0.5]),
    'neurons_layer_2': hp.choice('neurons_layer_2', [32, 64]),
    'dropout_2': hp.choice('dropout_2', [0.25, 0.5]),
    'neurons_layer_3': hp.choice('neurons_layer_3', [32, 64]),
    'dropout_3': hp.choice('dropout_3', [0.25, 0.5]),
    'neurons_layer_4': hp.choice('neurons_layer_4', [64, 128]),
    'dropout_4': hp.choice('dropout_4', [0.25, 0.5]),
    'neurons_layer_5': hp.choice('neurons_layer_5', [64, 128]),
    'dropout_5': hp.choice('dropout_5', [0.25, 0.5]),

    'activation': hp.choice('activation', ['relu']), # JUST relu for overfitting
    # 'leakyrelu_alpha': hp.choice('dropout_1', [0.05, 0.1]),
    'batch_size': hp.choice('batch_size', [64, 128, 512]), # reduce batch size space

}

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def dense_train(space):
    ''' train lightgbm booster based on training / validaton set -> give predictions of Y '''

    params = space.copy()
    print(params)

    model = models.Sequential()
    if params['activation'] == 'leakyrelu':     # try leaky relu
        for i in range(params['num_Dense_layer']):
            model.add(Dense(params['neurons_layer_{}'.format(i + 1)]))    # add layers accoridng to num_Dense_layer
            model.add(LeakyReLU(alpha=0.1))
            if params['dropout_{}'.format(i+1)] > 0:
                model.add(Dropout(params['dropout_{}'.format(i+1)]))
    else:
        for i in range(params['num_Dense_layer']):
            model.add(Dense(params['neurons_layer_{}'.format(i+1)], activation=params['activation']))
            if params['dropout_{}'.format(i+1)] > 0:
                model.add(Dropout(params['dropout_{}'.format(i+1)]))
    model.add(Dense(1))

    callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='auto')
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
    ''' train & evaluate each of the dense model '''

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
        pd.DataFrame(sql_result, index=[0]).to_sql('results_dense', con=conn, index=False, if_exists='append')
    engine.dispose()

    print('sql_result_before writing: ', sql_result)

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)
        hpot['best_history'] = history

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
        hpot['best_stock_df'].to_sql('results_dense_stock', con=conn, index=False, if_exists='append')
    engine.dispose()

    plot_history(hpot['history'])  # plot training history

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

    plt.savefig('results_dense/plot_dense_{}.png'.format(sql_result['trial_lgbm']))
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

    # default settings to
    exclude_fwd = False
    use_median = True
    chron_valid = False
    ibes_qcut_as_x = True
    sql_result['name'] = '5 layer'
    sql_result['y_type'] = 'ibes'

    # these are parameters used to load_data
    period_1 = dt.datetime(2013,3,31)
    qcut_q = 10
    sample_no = 25
    db_last_param, sql_result = read_db_last(sql_result, 'results_dense', first=True)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

    data = load_data()

    for add_ind_code in [0]: # 1 means add industry code as X
        data.split_entire(add_ind_code=add_ind_code)
        sql_result['icb_code'] = add_ind_code

        for i in tqdm(range(sample_no)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, qcut_q,
                                                                              y_type=sql_result['y_type'],
                                                                              exclude_fwd=exclude_fwd,
                                                                              use_median=use_median,
                                                                              chron_valid=chron_valid)

            print(feature_names)

            X_test = np.nan_to_num(sample_set['test_x'], nan=0)
            Y_test = sample_set['test_y']

            cv_number = 1
            for train_index, test_index in cv:
                sql_result['cv_number'] = cv_number

                X_train = np.nan_to_num(sample_set['train_x'][train_index], nan=0)
                Y_train = sample_set['train_y'][train_index]
                X_valid =  np.nan_to_num(sample_set['train_x'][test_index], nan=0)
                Y_valid = sample_set['train_y'][test_index]

                print(X_train.shape , Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)
                HPOT(space, 10)


