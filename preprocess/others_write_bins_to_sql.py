import datetime as dt
import numpy as np
import pandas as pd
from load_data_rnn import load_data
from dateutil.relativedelta import relativedelta
from sqlalchemy import create_engine, TIMESTAMP, TEXT, BIGINT, NUMERIC
from tqdm import tqdm
from miscel import check_dup

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

if __name__ == "__main__":
    ''' re-write cut_bins / med_train to table '''

    # training / testing sets split par
    indi_sectors = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010, 501010,
                   201020, 502030, 401010, 999999]  # icb_code with > 1300 samples + rests in single big model (999999)
    indi_industry_new = [11, 20, 30, 35, 40, 45, 51, 60, 65]
    indi_industry = [10, 15, 20, 30, 35, 40, 45, 50, 55, 60, 65]
    period_1 = dt.datetime(2013, 3, 31)     # starting point for first testing set

    # create dict storing values/df used in training
    sql_result = {}     # data write to DB TABLE lightgbm_results

    # parser
    sql_result['qcut_q'] = 10                           # number of Y classes
    sql_result['y_type'] = 'ibes'
    data = load_data()          # load all data: create load_data.main = df for all samples - within data(CLASS)

    bins_list = []

    total_test_id = {}
    for icb_code in ([0] + indi_industry_new): # + indi_sectors + [0, 1, 2]:   # roll over industries (first 2 icb code)
        # data.split_industry(icb_code, combine_ind=True)
        data.split_entire(add_ind_code=icb_code)
        sql_result['icb_code'] = icb_code

        total_test_id[icb_code] = 0
        for i in tqdm(range(21)):  # roll over testing period
            testing_period = period_1 + i * relativedelta(months=3)
            sql_result['testing_period'] = testing_period

            # sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period,
            #                                                                   sql_result['qcut_q'],
            #                                                                   sql_result['y_type'],
            #                                                                   use_median=True,
            #                                                                   chron_valid=False)

            train_x, train_y, X_test, Y_test, cv, test_id, x_col, cut_bins = data.split_train_test(testing_period,
                                                                                                   sql_result['qcut_q'],
                                                                                                   sql_result['y_type'],
                                                                                                   exclude_fwd=False,
                                                                                                   small_training=True,
                                                                                                   eps_only=False)

            total_test_id[icb_code] += len(test_id)
            for col in ['qcut_q', 'icb_code', 'testing_period', 'y_type']:
                cut_bins[col] = sql_result[col]
            for col in ['cut_bins', 'med_train']:
                cut_bins[col] = str(list(cut_bins[col]))

            cut_bins['label'] = 'rnn'
            bins_list.append(cut_bins)  # record cut_bins & median used in Y conversion
            # print(cut_bins)

    with engine.connect() as conn:  # record type of Y
        pd.DataFrame(bins_list).to_sql('results_bins_new', con=conn, index=False, if_exists='append', method='multi')
    engine.dispose()