from hyperopt import hp

space_big = {
    'num_Dense_layer': hp.choice('num_Dense_layer', [4, 5, 6]),  # number of layers ONE layer is TRIVIAL # drop 2, 3, 4
    'learning_rate': hp.choice('lr', [3, 4, 5]),    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
                                                    # remove lr = 5 & 7 after tuning
    'init_nodes': hp.choice('init_nodes', [16, 32]),  # nodes for Dense first layer -> LESS NODES
    'dropout': hp.choice('dropout', [0.25, 0.5]),

    'nodes_mult': hp.choice('nodes_mult', [0, 1]),          # nodes growth rate
    'mult_freq': hp.choice('mult_freq', [1, 2, 3]),         # nodes double frequency
    'mult_start': hp.choice('mult_start', [2, 3]),       # first layer nodes number growth

    'activation': hp.choice('activation', ['relu', 'tanh']), # JUST relu for overfitting
    'batch_size': hp.choice('batch_size', [128, 256]), # reduce batch size space # drop 512
}

space_small = {
    'num_Dense_layer': hp.choice('num_Dense_layer', [5, 6]),  # number of layers ONE layer is TRIVIAL # drop 2, 3, 4
    'learning_rate': hp.choice('lr', [4, 5]),    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
                                                    # remove lr = 5 & 7 after tuning
    'init_nodes': hp.choice('init_nodes', [4, 8]),  # nodes for Dense first layer -> LESS NODES
    'dropout': hp.choice('dropout', [0.1, 0.25]),

    'nodes_mult': hp.choice('nodes_mult', [0, 1]),          # nodes growth rate
    'mult_freq': hp.choice('mult_freq', [3]),         # nodes double frequency
    'mult_start': hp.choice('mult_start', [3]),       # first layer nodes number growth

    'activation': hp.choice('activation', ['relu',  'tanh']), # JUST relu for overfitting
    'batch_size': hp.choice('batch_size', [128, 256]), # reduce batch size space # drop 512
}

def find_hyperspace(sql_result):

    if 'big' in sql_result['name']:
        return space_big
    elif 'small' in sql_result['name']:
        return space_small
