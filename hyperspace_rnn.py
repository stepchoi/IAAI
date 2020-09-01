from hyperopt import hp

space_top = {
    'learning_rate': 2, # drop 7
    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
    'kernel_size': hp.choice('kernel_size',[64, 128]), #CNN kernel size - num of different "scenario"
    'num_gru_layer': hp.choice('num_gru_layer', [2, 3]),     # number of layers # drop 1, 2
    'gru_nodes_mult': hp.choice('gru_nodes_mult', [0, 1]),      # nodes growth rate *1 or *2
    'gru_nodes': hp.choice('gru_nodes', [4, 8]),    # start with possible 4 nodes -- 8, 8, 16 combination possible
    'gru_dropout': hp.choice('gru_drop', [0.25, 0.5]),

    'activation': 'tanh',
    'batch_size': hp.choice('batch_size', [128, 256]), # drop 1024
}

space_all = {
    'learning_rate': 3, # drop 7
    'kernel_size': hp.choice('kernel_size',[64, 384]), #CNN kernel size - num of different "scenario"
    'num_gru_layer': hp.choice('num_gru_layer', [2, 3]),     # number of layers # drop 1, 2
    'gru_nodes_mult': hp.choice('gru_nodes_mult', [0, 1]),      # nodes growth rate *1 or *2
    'gru_nodes': hp.choice('gru_nodes', [4, 8]),    # start with possible 4 nodes -- 8, 8, 16 combination possible
    'gru_dropout': hp.choice('gru_drop', [0.25, 0.5]),

    'activation': 'tanh',
    'batch_size': hp.choice('batch_size', [64]), # drop 1024
}

def find_hyperspace(sql_result):
    if 'top' in sql_result['name']:
        return space_top
    else:
        return space_all
