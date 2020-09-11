import os
import gc
import argparse
import datetime as dt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import r2_score, mean_absolute_error
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from load_data_rnn import load_data

from tensorflow.python.keras import callbacks, optimizers, initializers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GRU, Dropout, Flatten,  LeakyReLU, Input, Concatenate, Reshape, Lambda, Conv2D, Bidirectional
from tensorflow.python.keras import backend as K

import tensorflow as tf                             # avoid error in Tensorflow initialization
tf.compat.v1.disable_eager_execution()
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

space = {
    'learning_rate': 2,
    'num_gru_layer': hp.choice('num_gru_layer', [3, 4]),
    'gru_nodes_mult': hp.choice('gru_nodes_mult', [0, 1]),
    'gru_nodes': 1,
    'gru_dropout': hp.choice('gru_drop', [0.25, 0.5]),

    'num_dense_layer': hp.choice('num_dense_layer', [2, 3]),
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

    lookback = 20  # lookback = 5Y * 4Q = 20Q
    x_fields = 10  # lgbm top15 features -> 10 features in rnn

    inputs_loss_weight = 0.1  # loss weights for individual outputs from each rnn model
    dense_loss_weight = 2  # loss weights for final output
    loss_weights = [inputs_loss_weight] * x_fields + [dense_loss_weight]  # loss weights for training

    loss = ['mae'] * (x_fields + 1)  # use MAE loss function for all inputs and final
    metrics = ['mae'] * (x_fields + 1)

    input_img = Input(shape=(lookback, x_fields))
    layers = []
    for col in range(10):  # build model for each feature
        g_1 = K.expand_dims(input_img[:, :, col], axis=2)  # add dimension to certain feature: shape = (samples, 20, 1)

        for i in range(params['num_gru_layer']):
            temp_nodes = int(min(params['gru_nodes'] * (2 ** (params['gru_nodes_mult'] * i)), 8))  # nodes grow at 2X or stay same - at most 8 nodes
            extra = dict(return_sequences=True)
            if args.bi == False:
                if i == params['num_gru_layer'] - 1:
                    extra = dict(return_sequences=False)
                    g_2 = GRU(temp_nodes, **extra)(g_1)  # forecast state
                elif i == 0:
                    g_1 = GRU(temp_nodes, **extra)(g_1)
                else:
                    g_1 = GRU(temp_nodes, dropout=params['gru_dropout'], **extra)(g_1)
            else:  # try bidirectional one
                if i == params['num_gru_layer'] - 1:
                    extra = dict(return_sequences=False)
                    g_2 = GRU(temp_nodes, **extra)(g_1)  # forecast state
                elif i == 0:
                    g_1 = Bidirectional(GRU(temp_nodes, **extra))(g_1)
                else:
                    g_1 = Bidirectional(GRU(temp_nodes, dropout=params['gru_dropout'], **extra))(g_1)

        layers.append(g_2)

    f_x = Concatenate(axis=1)(layers)    # join all forecast states

    for i in range(params['num_dense_layer'] - 1):  # dense layers based on forecast states
        f_x = Dense(10)(f_x)

    f_x = Dense(1)(f_x)

    outputs = layers + [f_x]    # outputs = all forecast states + final dense
    model = Model(inputs=input_img, outputs=outputs)  # outputs = 10 forecast states + final forecast

    callbacks_list = [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
                      callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto')]  # add callbacks
    lr_val = 10 ** -int(params['learning_rate'])
    adam = optimizers.Adam(lr=lr_val)
    model.compile(adam, loss=loss, metrics=metrics, loss_weights=loss_weights)
    model.summary()

    history = model.fit(X_train, [Y_train] * (x_fields + 1), epochs=50, batch_size=params['batch_size'],
                        validation_data=(X_valid, [Y_valid] * (x_fields + 1)), verbose=1, callbacks=callbacks_list)

    Y_test_pred = model.predict(X_test)[-1]  # final dense predictions
    Y_train_pred = model.predict(X_train)[-1]
    Y_valid_pred = model.predict(X_valid)[-1]

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
    parser.add_argument('--exclude_fwd', default=False, action='store_true')       # using without I/B/E/S features
    args = parser.parse_args()
    gpu_mac_address(args)           # GPU set up

    period_1 = dt.datetime(2013,4,1)
    for i in tqdm(range(21)):            # roll over 21 testing period since 2013-Q1
        testing_period = period_1 + i * relativedelta(months=3) - relativedelta(days=1)
        train_x, train_y, X_test, Y_test, cv = load_data(testing_period, **vars(args))

        for train_index, test_index in cv:      # roll over 5-fold cross validation
            X_train = train_x[train_index]
            Y_train = train_y[train_index]
            X_valid = train_x[test_index]
            Y_valid = train_y[test_index]

            HPOT(space, 10)     # using X_train, Y_train, X_valid, Y_valid, X_test, Y_test for training & evaluation
            gc.collect()


