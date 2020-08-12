from hyperopt import hp

space_xgb = {
    'eta': hp.choice('eta', [0.01, 0.1]),
    'booster': hp.choice('boosting_type', ['gbtree', 'dart']),
    'gamma': hp.choice('gamma', [0, 0.01, 0.1]),
    'max_bin': hp.choice('max_bin', [128, 256]),
    'max_depth': hp.choice('max_depth', [8, 15, 20]),
    'num_leaves': hp.choice('num_leaves', [75, 125, 250]),
    'min_child_weight': hp.choice('min_child_weight', [1, 10, 20]),
    'colsample_bynode': hp.choice('feature_fraction', [0.7, 0.9]),
    'subsample': hp.choice('bagging_fraction', [0.8, 0.9]),
    'max_delta_step': hp.choice('max_delta_step', [1, 5, 10]),
    'alpha': hp.choice('lambda_l1', [0, 10]),
    'lambda': hp.choice('lambda_l2', [10, 100]),
    'grow_policy': 'lossguide'
}

def find_hyperspace(sql_result):

    return space_xgb

    # if sql_result['icb_code'] < 10:
    #     return space[0]
    # elif (sql_result['icb_code'] >= 10) and (sql_result['icb_code'] < 100):
    #     sp = space[sql_result['icb_code']]
    #     if 'mse' in sql_result['name']:
    #         sp.update({'min_gain_to_split': hp.choice('min_gain_to_split', [0, 0.001, 0.1]),
    #                    'lambda_l1': hp.choice('lambda_l1', [0, 1, 5]),
    #                    'lambda_l2': hp.choice('lambda_l2', [0, 1, 5]),
    #                    })
    #     return sp
    # elif sql_result['icb_code'] >= 100:
    #     sector_2_ind = {301010: 30, 101020: 11, 201030: 20, 302020: 30, 351020: 35, 502060: 51, 552010: 51, 651010: 65,
    #                     601010: 60, 502050: 51, 101010: 11, 501010: 51, 201020: 20, 502030: 51, 401010: 40, 999999: 0}
    #     print(sector_2_ind[sql_result['icb_code']])
    #     return space[sector_2_ind[sql_result['icb_code']]]

if __name__ == '__main__':
    sql_result = {'icb_code': 999}
    print(find_hyperspace(sql_result))