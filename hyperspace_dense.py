from hyperopt import hp

space = {}
space['big'] = {
    'num_Dense_layer': hp.choice('num_Dense_layer', [4, 5, 6]),  # number of layers ONE layer is TRIVIAL # drop 2, 3, 4
    'learning_rate': hp.choice('lr', [2, 3]),    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
                                                    # remove lr = 5 & 7 after tuning
    'init_nodes': hp.choice('init_nodes', [16, 32]),  # nodes for Dense first layer -> LESS NODES
    'end_nodes': 64,
    'dropout': hp.choice('dropout', [0.25, 0.5]),

    'nodes_mult': hp.choice('nodes_mult', [0, 1]),          # nodes growth rate
    'mult_freq': hp.choice('mult_freq', [1, 2]),         # nodes double frequency
    'mult_start': 2,       # first layer nodes number growth

    'activation': 'tanh', # JUST relu for overfitting
    'batch_size': 128, # reduce batch size space # drop 512
}

space['small'] = {
    'num_Dense_layer': hp.choice('num_Dense_layer', [3, 4, 5]),  # number of layers ONE layer is TRIVIAL # drop 2, 3, 4
    'learning_rate': 2,    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
                                                    # remove lr = 5 & 7 after tuning
    'init_nodes': hp.choice('init_nodes', [8, 16]),  # nodes for Dense first layer -> LESS NODES
    'dropout': hp.choice('dropout', [0.5, 0.25]),
    'end_nodes':32,

    'nodes_mult': hp.choice('nodes_mult', [0, 1]),          # nodes growth rate
    'mult_freq': hp.choice('mult_freq', [2, 3]),         # nodes double frequency
    'mult_start': 2,       # first layer nodes number growth

    'activation': 'tanh', # JUST relu for overfitting
    'batch_size': 128, # reduce batch size space # drop 512
}

space['top15'] = {
    'num_Dense_layer': hp.choice('num_Dense_layer',[4, 5, 6]),
    'learning_rate': 2,    # => 1e-x - learning rate - REDUCE space later - correlated to batch size
                                                    # remove lr = 5 & 7 after tuning
    'init_nodes': hp.choice('init_nodes',[8, 16]),  # nodes for Dense first layer -> LESS NODES
    'dropout': 0,
    'end_nodes': 32,

    'nodes_mult':  hp.choice('nodes_mult',[0,1]),       # nodes growth rate
    'mult_freq': 3,         # nodes double frequency
    'mult_start': 2,      # first layer nodes number growth

    'activation':'tanh',  # JUST relu for overfitting
    'batch_size': 128 # reduce batch size space # drop 512
}

def find_hyperspace(args):

    return space[args.hyperspace_type]
