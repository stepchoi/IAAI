from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
from preprocess.ratios import full_period, worldscope
from miscel import date_type, check_dup
from collections import Counter
import os

from results_lgbm.consensus import yoy_to_median, eps_to_yoy, download_add_detail

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def download_ibes_median():
    ''' Download ibes_data and organize to YoY and convert to qcut median '''

    try:
        yoy_med = pd.read_csv('results_lgbm/compare_with_ibes/ibes_median.csv')
        yoy_med = date_type(yoy_med)
        print('local version run - ibes_median')
    except:
        yoy = eps_to_yoy().merge_and_calc()
        yoy_med = yoy_to_median(yoy)  # STEP2: convert ibes YoY to qcut / median
        yoy_med.to_csv('results_lgbm/compare_with_ibes/ibes_median.csv', index=False)

    return yoy_med

def download_stock():
    ''' Download results_dense2_stock for stocks '''

    try:
        detail_stock = pd.read_csv('results_lgbm/compare_with_ibes/dense_stock_{}.csv'.format(r_name))
        detail_stock = date_type(detail_stock, date_col='testing_period')
        print('local version run - stock_{}'.format(r_name))
    except:
        print('----------------> update stock results from DB TABLE results_dense2')

        with engine.connect() as conn:
            # read DB TABLE results_lightgbm data for given "name"
            result_all = pd.read_sql(
                "SELECT trial_lgbm, icb_code, testing_period, cv_number, mae_test, y_type "
                "FROM results_dense2 WHERE name='{}'".format(r_name), conn)
            trial_lgbm = set(result_all['trial_lgbm'])

            # read corresponding part of DB TABLE results_lightgbm_stock
            query = text('SELECT * FROM results_dense2_stock WHERE (trial_lgbm IN :trial_lgbm)')
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_stock = pd.read_sql(query, conn)
        engine.dispose()

        detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')  # map training information to stock data
        detail_stock['exclude_fwd'] = True
        detail_stock['qcut_q'] = 10

        detail_stock.to_csv('results_lgbm/compare_with_ibes/dense_stock_{}.csv'.format(r_name), index=False)

    print(detail_stock)

    return detail_stock


if __name__ == "__main__":

    r_name = 'new'

    download_stock()