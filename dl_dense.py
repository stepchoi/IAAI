import datetime as dt
from dateutil.relativedelta import relativedelta
import argparse
from tqdm import tqdm
from hyperopt import fmin, tpe, STATUS_OK, Trials
import os

from load_data_dense import load_data
from hyperspace_dense import find_hyperspace

from tensorflow.python.keras import callbacks, optimizers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras import backend as K
from sklearn.metrics import r2_score, mean_absolute_error

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

    init_nodes = params['init_nodes']       # fisrt dense layer - number of nodes
    nodes_mult = params['nodes_mult']       # nodes growth rate
    mult_freq = params['mult_freq']         # grow every X layer
    mult_start = params['mult_start']       # grow from X layer
    end_nodes = params['end_nodes']         # maximum number of nodes

    if params['num_Dense_layer'] < 4:
        params['init_nodes'] = init_nodes = 16

    d_1 = Dense(init_nodes, activation=params['activation'])(input_img)  # remove kernel_regularizer=regularizers.l1(params['l1'])
    d_1 = Dropout(params['dropout'])(d_1)

    for i in range(1, params['num_Dense_layer']):
        temp_nodes = int(min(init_nodes * (2 ** (nodes_mult * max((i - mult_start + 3)//mult_freq, 0))), end_nodes))
        d_1 = Dense(temp_nodes, activation=params['activation'])(d_1)

        if i != params['num_Dense_layer'] - 1:    # last dense layer has no dropout
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

    parser = argparse.ArgumentParser()      # setting for different configuration
    parser.add_argument('--objective', default='regression_l1')     # regression_l2 = optimizing mse
    parser.add_argument('--exclude_fwd', default=False, action='store_true')     # True = no I/B/E/S consensus features
    parser.add_argument('--sample_type', default='industry')
    parser.add_argument('--sample_ratio', default=1, type=float) # 0.5 = select 50% random sample from original data set
    args = parser.parse_args()

    if args.sample_type == 'industry':  # config III
        partitions = [11, 20, 30, 35, 40, 45, 51, 60, 65]   # 11 represents industry (10 + 15); 51 represents (50 + 55)
    elif args.sample_type == 'entire':  # config II
        partitions = [0]    # 0 represents aggregate model

    period_1 = dt.datetime(2013, 4, 1)     # starting point for first testing set

    for icb_code in partitions:   # roll over industries (first 2 icb code)
        for i in tqdm(range(21)):  # roll over 2013-3-31 to 2018-3-31
            testing_period = period_1 + i * relativedelta(months=3) - relativedelta(days=1)
            train_x, train_y, X_test, Y_test, cv, feature_names = load_data(testing_period, **vars(args))

            for train_index, test_index in cv:     # roll over 5 cross validation set
                X_train = train_x[train_index]
                Y_train = train_y[train_index]
                X_valid = train_x[test_index]
                Y_valid = train_y[test_index]

                space = find_hyperspace(args)
                HPOT(space, max_evals=10)   # start hyperopt



