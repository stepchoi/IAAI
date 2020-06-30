import json
from multiprocessing import cpu_count

import lightgbm as lgb
import numpy as np
from hyperopt import fmin, tpe, hp
from sklearn.metrics import mean_absolute_error
from sqlalchemy import create_engine
from sqlalchemy.sql import text

HYPEROPT_EVALS = 20
eval_run = 0


def get_stats():
    arya_db_url = "postgres://postgres:askLORA20$@aryalite-1.crpzxxbaqqm2.ap-northeast-2.rds.amazonaws.com:5432/postgres"
    arya_db_engine = create_engine(arya_db_url, pool_size=cpu_count(), max_overflow=-1, isolation_level="AUTOCOMMIT")
    production_db_url = "postgres://postgres:askLORA20$@seoul-prod2.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres"
    production_db_engine = create_engine(production_db_url, pool_size=cpu_count(), max_overflow=-1,
                                         isolation_level="AUTOCOMMIT")

    # get rics, start and end dates from universe in arya
    get_universe_sql = text('select ric, start_date,end_date FROM universe')
    with arya_db_engine.connect() as conn:
        get_universe = conn.execute(get_universe_sql).fetchall()

    # for each ric
    for universe_entry in get_universe:
        ric = universe_entry[0]
        start_date = universe_entry[1]
        end_date = universe_entry[2]

        # get the fundamentals stats for that start and end
        get_fundamental_statistics_sql = text(
            'SELECT :ric,earliest_date, latest_date, s1.number_of_quarters, s1.max_possible_number_of_quarters, s1.latest_quarter_with_all_data, s1.earliest_quarter_with_all_data FROM ( SELECT min(fs.start_date) AS earliest_date, max(fs.end_date) AS latest_date FROM fundamentals_statistics fs WHERE fs.start_date >= :start_date AND fs.end_date <= :end_date AND fs.ric=:ric ) s, LATERAL ( SELECT fs.number_of_days, fs.max_possible_number_of_days, fs.number_of_quarters, fs.max_possible_number_of_quarters, fs.latest_quarter_with_all_data, fs.earliest_quarter_with_all_data FROM fundamentals_statistics fs WHERE fs.ric=:ric AND fs.start_date = s.earliest_date AND fs.end_date = s.latest_date) s1')
        with arya_db_engine.connect() as conn:
            get_fundamental_statistics = conn.execute(get_fundamental_statistics_sql, ric=ric, start_date=start_date,
                                                      end_date=end_date).fetchall()
        print(get_fundamental_statistics)
        # get skipped quarters
        get_missing_quarters_sql = text('SELECT ric, missing_quarter_list FROM missing_quarters WHERE ric=:ric')
        with arya_db_engine.connect() as conn:
            get_missing_quarters = conn.execute(get_missing_quarters_sql, ric=ric).fetchall()
        print(get_missing_quarters)
        # get close stats for that start and end
        get_close_statistics_sql = text(
            'SELECT ric, fs.number_of_days, fs.max_possible_number_of_days, fs.latest_day_with_all_data, fs.earliest_day_with_all_data FROM close_statistics fs WHERE fs.ric=:ric AND fs.start_date = :start_date AND fs.end_date = :end_date')
        with production_db_engine.connect() as conn:
            get_close_statistics = conn.execute(get_close_statistics_sql, ric=ric, start_date=start_date,
                                                end_date=end_date).fetchall()
        print(get_close_statistics)
        # get missing days days
        get_missing_days_sql = text('SELECT :ric, missing_days_list FROM missing_days WHERE ticker=:ric')
        with production_db_engine.connect() as conn:
            get_missing_days = conn.execute(get_missing_days_sql, ric=ric).fetchall()
        print(get_missing_days)


def get_lgbm_input():
    arya_db_url = "postgres://postgres:askLORA20$@aryalite-1.crpzxxbaqqm2.ap-northeast-2.rds.amazonaws.com:5432/postgres"
    arya_db_engine = create_engine(arya_db_url, pool_size=cpu_count(), max_overflow=-1, isolation_level="AUTOCOMMIT")
    #    get_lgbm_input_sql =text('SELECT vix_open, vix_high, vix_low, vix_close, ni_over_mc, op_over_mc, ns_over_mc, days_21_c2c_vols, days_21_c2c_vols_before_21_days, days_21_c2c_vols_before_42_days, days_63_c2c_vols_before_63_days, days_125_c2c_vols_before_125_days, days_256_c2c_vols_before_256_days, days_21_roger_satchell_vols, days_21_roger_satchell_vols_before_21_days, days_21_roger_satchell_vols_before_42_days, days_63_roger_satchell_vols_before_63_days, days_125_roger_satchell_vols_before_125_days, days_256_roger_satchell_vols_before_256_days, days_125_kurtosis, days_125_kurtosis_before_125_days, days_256_kurtosis_before_256_days, days_21_total_return, days_63_total_return, days_125_total_return_before_21_days, days_235_total_return_before_21_days, days_125_skew, days_125_skew_before_125_days, total_returns_1, total_returns_5 FROM zz_test_simple_vols_forecast as o JOIN lgbm_price_input as i ON i.ric=o.ric and i.trading_day=o.trading_day ORDER BY o.ric, o.trading_day')
    get_lgbm_input_sql = text(
        'SELECT vix_close, days_21_c2c_vols, days_21_c2c_vols_before_21_days, days_21_c2c_vols_before_42_days, days_63_c2c_vols_before_63_days, days_125_c2c_vols_before_125_days, days_256_c2c_vols_before_256_days, days_21_roger_satchell_vols, days_21_roger_satchell_vols_before_21_days, days_21_roger_satchell_vols_before_42_days, days_63_roger_satchell_vols_before_63_days, days_125_roger_satchell_vols_before_125_days, days_256_roger_satchell_vols_before_256_days, days_256_kurtosis_before_256_days, days_63_total_return, days_125_total_return_before_21_days, days_235_total_return_before_21_days, total_returns_1 FROM zz_test_simple_vols_forecast as o JOIN lgbm_price_input as i ON i.ric=o.ric and i.trading_day=o.trading_day ORDER BY o.ric, o.trading_day')

    with arya_db_engine.connect() as conn:
        lgbm_input = conn.execute(get_lgbm_input_sql).fetchall()

    lgbm_input = np.asfarray(lgbm_input)
    return lgbm_input


def get_lgbm_output(output_type):
    arya_db_url = "postgres://postgres:askLORA20$@aryalite-1.crpzxxbaqqm2.ap-northeast-2.rds.amazonaws.com:5432/postgres"
    arya_db_engine = create_engine(arya_db_url, pool_size=cpu_count(), max_overflow=-1, isolation_level="AUTOCOMMIT")
    #    get_lgbm_output_sql =text('SELECT v21, v63, v125 FROM zz_test_simple_vols_forecast as o JOIN lgbm_input as i ON i.ric=o.ric and i.trading_day=o.trading_day ORDER BY o.ric, o.trading_day')
    if output_type == 'v21':
        get_lgbm_output_sql = text(
            'SELECT o.ric, o.trading_day, v21 FROM zz_test_simple_vols_forecast as o JOIN lgbm_price_input as i ON i.ric=o.ric and i.trading_day=o.trading_day ORDER BY o.ric, o.trading_day')
    elif output_type == 'v63':
        get_lgbm_output_sql = text(
            'SELECT o.ric, o.trading_day, v63 FROM zz_test_simple_vols_forecast as o JOIN lgbm_price_input as i ON i.ric=o.ric and i.trading_day=o.trading_day ORDER BY o.ric, o.trading_day')
    else:
        get_lgbm_output_sql = text(
            'SELECT o.ric, o.trading_day, v125 FROM zz_test_simple_vols_forecast as o JOIN lgbm_price_input as i ON i.ric=o.ric and i.trading_day=o.trading_day ORDER BY o.ric, o.trading_day')

    with arya_db_engine.connect() as conn:
        lgbm_output = conn.execute(get_lgbm_output_sql).fetchall()

    #    lgbm_output = np.asfarray(lgbm_output)
    return lgbm_output


def save_predictions(hypers, vol_type, accuracy, vol_predictions, test_error):
    arya_db_url = "postgres://postgres:askLORA20$@aryalite-1.crpzxxbaqqm2.ap-northeast-2.rds.amazonaws.com:5432/postgres"
    arya_db_engine = create_engine(arya_db_url, pool_size=cpu_count(), max_overflow=-1, isolation_level="AUTOCOMMIT")

    insert_predictions_sql = text(
        'SELECT store_lgm_results(CAST(:hypers as json), CAST(:vol_type AS text), CAST(:accuracy AS double precision), CAST(:vol_predictions AS vol_prediction[]), CAST(:test_error as double precision))')
    with arya_db_engine.connect() as conn:
        conn.execute(insert_predictions_sql, hypers=hypers, vol_type=vol_type, accuracy=accuracy,
                     vol_predictions=vol_predictions, test_error=test_error)


def run_model(X_train, X_test, X_valid, Y_test, Y_train, Y_valid, dates_and_tickers_train, dates_and_tickers_test,
              vol_type):
    number_of_evaluations_already_run = 0

    # Y_train = np.power(Y_train,2)
    def inner_fn(hypers):
        nonlocal number_of_evaluations_already_run

        lgb_train = lgb.Dataset(X_train, Y_train)
        lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

        print('Starting training...')
        # train
        gbm = lgb.train(hypers,
                        lgb_train,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=30)

        print('Starting validating...')
        # validate
        y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
        # eval
        error = mean_absolute_error(Y_valid, y_pred)
        # error = np.power(mean_absolute_error(Y_valid, y_pred),0.5)
        print('The l1 error of prediction is:', error)

        # predict
        print('Starting predicting...')
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        test_error = mean_absolute_error(Y_test, y_pred)
        # merge the array of dates and tickers, test data and prediction
        data = []
        for i in range(0, len(y_pred)):
            data.append(((dates_and_tickers_test[i])[0], (dates_and_tickers_test[i])[1], Y_test[i], y_pred[i]))

        print(data)
        # save it in the db
        save_predictions(json.dumps(hypers, default=str), vol_type, error, data, test_error)
        # currentDT = datetime.datetime.now()
        # gbm.save_model(str(currentDT) + '_model.txt', num_iteration=gbm.best_iteration)
        # shap_values = shap.TreeExplainer(gbm).shap_values(X_train)
        # shap.summary_plot(shap_values, X_train)
        yy = number_of_evaluations_already_run
        return error

    number_of_evaluations_already_run += 1

    return inner_fn


if __name__ == '__main__':
    # get_stats()

    vol_type = 'v63'
    # get data
    x = get_lgbm_input()
    x2 = x
    all_output = get_lgbm_output(vol_type)
    array_to_shuffle = []
    for i in range(0, len(x)):
        array_to_shuffle.append((x[i], all_output[i]))

    np.random.shuffle(array_to_shuffle)
    x = np.asfarray([z[0] for z in array_to_shuffle])
    all_output = [z[1] for z in array_to_shuffle]
    dates_and_tickers = [(z[0], z[1]) for z in all_output]
    y = np.asfarray([z[2] for z in all_output])
    y = y.reshape((len(y)))

    # strip out the nans from both input and output
    nan_x = np.isnan(x)
    nan_x1 = np.logical_not(nan_x)
    nan_x2 = np.logical_and.reduce(nan_x1, axis=1)
    nan_y = np.isnan(y)
    nan_y1 = np.logical_not(nan_y)
    nan_xy = np.logical_and(nan_x2, nan_y1)
    non_nan_x = x[nan_xy]
    non_nan_y = y[nan_xy]

    avgs = np.mean(non_nan_y, dtype=np.float32)
    stdevs = np.std(non_nan_y, dtype=np.float32)
    # cutoff outliers at +1 stdevs
    non_nan_y = np.minimum(non_nan_y, avgs + 1 * stdevs)
    non_nan_y = np.divide(non_nan_y, 5)
    non_nan_y = np.round(non_nan_y, 2)
    non_nan_y = np.multiply(non_nan_y, 5)

    non_nan_dates_and_tickers = [i for indx, i in enumerate(dates_and_tickers) if nan_xy[indx] == True]

    # get training and test set
    offset1 = int(len(non_nan_x) * 0.6)
    offset2 = int(len(non_nan_x) * 0.8)

    X_train = non_nan_x[0:offset1]
    X_valid = non_nan_x[offset1:offset2]
    X_test = non_nan_x[offset2:]
    Y_train = non_nan_y[0:offset1]
    Y_valid = non_nan_y[offset1:offset2]
    Y_test = non_nan_y[offset2:]
    dates_and_tickers_train = non_nan_dates_and_tickers[0:offset1]
    dates_and_tickers_valid = non_nan_dates_and_tickers[offset1:offset2]
    dates_and_tickers_test = non_nan_dates_and_tickers[offset2:]

    space = {
        # better accuracy
        'learning_rate': hp.choice('learning_rate', [0.05, 0.1, 0.2, 0.3]),
        'boosting_type': 'dart',  # past:  hp.choice('boosting_type', ['gbdt', 'dart']
        'max_bin': hp.choice('max_bin', [75, 100, 150, 255]),
        'num_leaves': hp.choice('num_leaves', np.arange(25, 200, 25, dtype=int)),  # avoid overfit
        'min_data_in_leaf': hp.choice('min_data_in_leaf', np.arange(25, 175, 50, dtype=int)),
        'feature_fraction': hp.choice('feature_fraction', np.arange(0.70, 0.95, 0.05)),
        'bagging_fraction': hp.choice('bagging_fraction', np.arange(0.5, 0.7, 0.05)),
        'bagging_freq': hp.choice('bagging_freq', [2, 4, 8]),
        'min_gain_to_split': hp.choice('min_gain_to_split', np.arange(0.001, 0.05, 0.003)),
        # 'lambda_l1': hp.choice('lambda_l1', [0, 0.5, 2, 10]),
        # 'lambda_l2': hp.choice('lambda_l2', [0, 0.5, 2, 10]),  # parameters won't change
        'objective': 'regression',
        'metric': 'l1',
        'type': 'regression_l1',
        'num_boost_round': 1000,
        'num_threads': cpu_count()
    }

    best = fmin(
        run_model(X_train, X_test, X_valid, Y_test, Y_train, Y_valid, dates_and_tickers_train, dates_and_tickers_test,
                  vol_type), space=space, algo=tpe.suggest, max_evals=HYPEROPT_EVALS)
