import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import ast
import argparse

import os
from hyperopt import fmin, tpe, STATUS_OK, Trials
from tensorflow.python.keras import callbacks, optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras import backend as K
from sklearn.metrics import r2_score, mean_absolute_error

from sqlalchemy import create_engine
from tqdm import tqdm

from load_data_lgbm import load_data
from hyperspace_dense import find_hyperspace
import matplotlib.pyplot as plt

import tensorflow as tf                             # avoid error in Tensorflow initialization
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

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
    print(params)

    input_shape = (X_train.shape[-1],)      # input shape depends on x_fields used
    input_img = Input(shape=input_shape)

    if 'fix' in sql_result['name']:     # run with defined model structure e.g. [8, 8, 8]
        nodes_list = ast.literal_eval(params['num_nodes'])  # convert str to nested dictionary
        print(nodes_list)

        d_1 = Dense(nodes_list[0], activation=params['activation'])(input_img)
        d_1 = Dropout(params['dropout'])(d_1)
        for i in range(1, len(nodes_list)):
            temp_nodes = nodes_list[i]
            d_1 = Dense(temp_nodes, activation=params['activation'])(d_1)
            if i != len(nodes_list) - 1:    # last dense layer has no dropout
                d_1 = Dropout(params['dropout'])(d_1)

        sql_result['num_nodes'] = str(nodes_list)

    else:
        init_nodes = params['init_nodes']
        nodes_mult = params['nodes_mult']
        mult_freq = params['mult_freq']
        mult_start = params['mult_start']
        num_Dense_layer = params['num_Dense_layer']

        if num_Dense_layer < 4:
            params['init_nodes'] = init_nodes = 16

        # if nodes_mult == 1:
        #     sql_result['num_Dense_layer'] = num_Dense_layer = int(np.log2(params['end_nodes']) - np.log2(params['init_nodes'])) + 1
        # elif nodes_mult == 0:
        #     sql_result['num_Dense_layer'] = num_Dense_layer = 5

        d_1 = Dense(init_nodes, activation=params['activation'])(input_img)  # remove kernel_regularizer=regularizers.l1(params['l1'])
        d_1 = Dropout(params['dropout'])(d_1)

        nodes = [init_nodes]
        for i in range(1, num_Dense_layer):
            # temp_nodes = nodes_list[i]
            temp_nodes = int(min(init_nodes * (2 ** (nodes_mult * max((i - mult_start + 3)//mult_freq, 0))), params['end_nodes'])) # nodes grow at 2X or stay same - at most 128 nodes
            d_1 = Dense(temp_nodes, activation=params['activation'])(d_1)  # remove kernel_regularizer=regularizers.l1(params['l1'])
            nodes.append(temp_nodes)

            if i != num_Dense_layer - 1:    # last dense layer has no dropout
                d_1 = Dropout(params['dropout'])(d_1)

        print(nodes)
        sql_result['num_nodes'] = str(nodes)

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

    # train_mae = model.evaluate(X_train, Y_train,  verbose=1)
    # valid_mae = model.evaluate(X_valid, Y_valid, verbose=1)
    # test_mae = model.evaluate(X_test, Y_test, verbose=1)
    Y_test_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)
    Y_valid_pred = model.predict(X_valid)

    return Y_test_pred, Y_train_pred, Y_valid_pred, history

def eval(space):
    ''' train & evaluate each of the dense model '''

    Y_test_pred, Y_train_pred, Y_valid_pred, history = dense_train(space)

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

    if result['mae_valid'] < hpot['best_mae']:  # update best_mae to the lowest value for Hyperopt
        hpot['best_mae'] = result['mae_valid']
        hpot['best_stock_df'] = pred_to_sql(Y_test_pred)
        hpot['best_history'] = history
        hpot['best_trial'] = sql_result['trial_lgbm']

    K.clear_session()
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
        pd.DataFrame(hpot['all_results']).to_sql('results_dense2', con=conn, index=False, if_exists='append', method='multi')
        hpot['best_stock_df'].to_sql('results_dense2_stock', con=conn, index=False, if_exists='append', method='multi')
    engine.dispose()

    # plot_history(hpot['best_history'])  # plot training history

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

    plt.savefig('results_dense/plot_dense_{}_{}.png'.format(hpot['best_trial'], hpot['best_mae']))
    plt.close()

def pred_to_sql(Y_test_pred):
    ''' prepare array Y_test_pred to DataFrame ready to write to SQL '''

    df = pd.DataFrame()
    df['identifier'] = test_id
    df['pred'] = Y_test_pred
    df['trial_lgbm'] = [sql_result['trial_lgbm']] * len(test_id)
    df['name'] = [sql_result['name']] * len(test_id)
    # print('stock-wise prediction: ', df)

    return df

def read_db_last(sql_result, results_table = 'results_dense2'):
    ''' read last records on DB TABLE lightgbm_results for resume / trial_no counting '''

    try:
        with engine.connect() as conn:
            db_last = pd.read_sql("SELECT * FROM {} Order by trial_lgbm desc LIMIT 1".format(results_table), conn)
        engine.dispose()

        db_last_param = db_last[['icb_code','testing_period']].to_dict('index')[0]
        db_last_trial_hpot = int(db_last['trial_hpot'])
        db_last_trial_lgbm = int(db_last['trial_lgbm'])

        sql_result['trial_hpot'] = db_last_trial_hpot + args.trial_lgbm_add  # trial_hpot = # of Hyperopt performed (n trials each)
        sql_result['trial_lgbm'] = db_last_trial_lgbm + args.trial_lgbm_add  # trial_lgbm = # of Lightgbm performed
        print('if resume from: ', db_last_param,'; sql last trial_lgbm: ', sql_result['trial_lgbm'])
    except:
        db_last_param = None
        sql_result['trial_hpot'] = sql_result['trial_lgbm'] = 0

    return db_last_param, sql_result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name_sql', required=True)
    parser.add_argument('--objective', default='regression_l1')
    parser.add_argument('--exclude_stock', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--exclude_fwd', default=False, action='store_true')
    parser.add_argument('--sample_type', default='entire')
    parser.add_argument('--y_type', default='ibes')
    parser.add_argument('--sample_no', type=int, default=21)
    parser.add_argument('--qcut_q', default=10, type=int)
    parser.add_argument('--trial_lgbm_add', default=1, type=int)
    parser.add_argument('--sample_ratio', default=1, type=float)
    parser.add_argument('--nthread', default=12, type=int)
    parser.add_argument('--sleep', type=int, default=0)
    parser.add_argument('--gpu_number', type=int, default=1)
    args = parser.parse_args()

    gpu_mac_address(args)

    from time import sleep
    sleep(args.sleep)

    # training / testing sets split par
    market_list = ['normal'] # default setting = all samples cross countries
    if args.sample_type == 'industry':
        partitions = [11, 20, 30, 35, 40, 45, 51, 60, 65]
    elif args.sample_type == 'sector':
        partitions = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                       201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    elif args.sample_type == 'entire':
        partitions = [0]
    elif args.sample_type == 'market':
        partitions = [0]
        market_list = ['us', 'jp', 'cn', 'hk']


    period_1 = dt.datetime(2013, 3, 31)     # starting point for first testing set
    base_space = {'verbose': -1,
                  'num_threads': args.nthread}  # for the best speed, set this to the number of real CPU cores

    # create dict storing values/df used in training
    sql_result = {}     # data write to DB TABLE lightgbm_results
    hpot = {}           # storing data for best trials in each Hyperopt
    resume = args.resume  # change to True if want to resume from the last running as on DB TABLE lightgbm_results
    sample_no = args.sample_no  # number of training/testing period go over ( 25 = until 2019-3-31)

    load_data_params = {'exclude_fwd': args.exclude_fwd,
                        'use_median': True,
                        'chron_valid': False,
                        'y_type': args.y_type,
                        'qcut_q': args.qcut_q,
                        'ibes_qcut_as_x': not(args.exclude_fwd),
                        'exclude_stock': args.exclude_stock,
                        }

    for mkt in market_list:     # roll over partition for each market (for IIIb)
        data = load_data(macro_monthly=True, market=mkt, sample_ratio=args.sample_ratio)          # load all data: create load_data.main = df for all samples - within data(CLASS)
        sql_result['market'] = mkt

        # sql_result['objective'] = base_space['objective'] = args.objective
        # x_type_map = {True: 'fwdepsqcut', False: 'ni'} # True/False based on exclude_fwd
        # sql_result['x_type'] = x_type_map[args.exclude_fwd]
        sql_result['name'] = args.name_sql

        # update load_data data
        # sql_result['qcut_q'] = load_data_params['qcut_q']     # number of Y classes
        sql_result['y_type'] = load_data_params['y_type']

        ''' start roll over testing period(25) / icb_code(16) / cross-validation sets(5) for hyperopt '''

        db_last_param, sql_result = read_db_last(sql_result)  # update sql_result['trial_hpot'/'trial_lgbm'] & got params for resume (if True)

        for icb_code in partitions:   # roll over industries (first 2 icb code)

            data.split_industry(icb_code, combine_ind=True)
            sql_result['icb_code'] = icb_code

            for i in tqdm(range(sample_no)):  # roll over testing period
                testing_period = period_1 + i * relativedelta(months=3)
                sql_result['testing_period'] = testing_period

                # when setting resume = TRUE -> continue training from last records in DB results_lightgbm
                if resume == True:

                    if {'icb_code': icb_code, 'testing_period': pd.Timestamp(testing_period)} == db_last_param:  # if current loop = last records
                        resume = False
                        print('---------> Resume Training', icb_code, testing_period)
                    else:
                        print('Not yet resume: params done', icb_code, testing_period)
                        continue

                # try:
                sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, **load_data_params)
                sql_result['exclude_fwd'] = load_data_params['exclude_fwd']

                # print('23355L106' in test_id)
                print(feature_names)
                if 'i0eps' in feature_names:
                    NameError('WRONG feature_names with i0eps!')

                # to_sql_bins(cut_bins)   # record cut_bins & median used in Y conversion

                X_test = np.nan_to_num(sample_set['test_x'], nan=0)
                Y_test = sample_set['test_y']

                sql_result['number_features'] = X_test.shape[1]

                cv_number = 1
                for train_index, test_index in cv:
                    sql_result['cv_number'] = cv_number

                    X_train = np.nan_to_num(sample_set['train_x'][train_index], nan=0)
                    Y_train = sample_set['train_y'][train_index]
                    X_valid = np.nan_to_num(sample_set['train_x'][test_index], nan=0)
                    Y_valid = sample_set['train_y'][test_index]

                    print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)
                    space = find_hyperspace(sql_result)
                    HPOT(space, 10)

                    sql_result['trial_hpot'] += 1
                    cv_number += 1



