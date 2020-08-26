from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
from preprocess.x_ratios import full_period, worldscope
from miscel import date_type, check_dup
from collections import Counter
import os

from results_analysis.lgbm_consensus import yoy_to_median, eps_to_yoy, label_sector, calc_mae_write, combine

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def download_ibes_median():
    ''' Download ibes_data and organize to YoY and convert to qcut median '''

    try:
        yoy_med = pd.read_csv('results_analysis/compare_with_ibes/ibes_median.csv')
        yoy_med = date_type(yoy_med)
        print('local version run - ibes_median')
    except:
        yoy = eps_to_yoy().merge_and_calc()
        yoy_med = yoy_to_median(yoy)  # STEP2: convert ibes YoY to qcut / median
        yoy_med.to_csv('results_analysis/compare_with_ibes/ibes_median.csv', index=False)

    return yoy_med

def download_stock():
    ''' Download results_dense2_stock for stocks '''

    try:
        detail_stock = pd.read_csv('results_analysis/compare_with_ibes/dense_stock_{}.csv'.format(r_name))
        detail_stock = date_type(detail_stock, date_col='testing_period')
        print('local version run - stock_{}'.format(r_name))
    except:
        print('----------------> update stock results from DB TABLE results_dense2')

        with engine.connect() as conn:
            # read DB TABLE results_lightgbm data for given "name"
            result_all = pd.read_sql(
                "SELECT name, trial_lgbm, icb_code, testing_period, cv_number, mae_test, y_type, number_features "
                "FROM results_dense2 WHERE name='{}'".format(r_name), conn)
            trial_lgbm = set(result_all['trial_lgbm'])

            # read corresponding part of DB TABLE results_lightgbm_stock
            query = text('SELECT * FROM results_dense2_stock WHERE (trial_lgbm IN :trial_lgbm)')
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_stock = pd.read_sql(query, conn)
        engine.dispose()

        detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')  # map training information to stock data
        detail_stock.to_csv('results_analysis/compare_with_ibes/dense_stock_{}.csv'.format(r_name), index=False)
        print('detial_stock shape: ', detail_stock)

    print(detail_stock)

    return detail_stock


def merge_ibes_stock(sp_only):
    ''' merge ibes and detail stock data '''

    yoy_med = download_ibes_median()
    detail_stock = download_stock()

    if sp_only==True:
        detail_stock['label'] = 'lgbm_sp'
    else:
        detail_stock['label'] = 'lgbm_normal'

    detail_stock['x_type'] = 'fwdepsqcut'
    detail_stock['number_features'] = 46

    detail_stock = detail_stock.drop_duplicates(subset=['icb_code', 'identifier', 'testing_period', 'cv_number',
                                                        'y_type', 'x_type', 'number_features'], keep='last')

    if not 'industry' in r_name:
        # detail_stock.loc[detail_stock['icb_code'] == 1, 'x_type'] = 'fwdepsqcut-industry_code'
        # detail_stock.loc[detail_stock['icb_code'] == 2, 'x_type'] = 'fwdepsqcut-sector_code'
        detail_stock = detail_stock.loc[detail_stock['icb_code']==0]

    feature_list = list(set(detail_stock['number_features'].dropna().to_list()))

    for i in feature_list:
        detail_stock.loc[detail_stock['number_features'] == i, 'x_type'] += '-{}'.format(i)

    print(list(set(detail_stock['x_type'].dropna().to_list())))

    detail_stock = detail_stock.drop_duplicates(subset=['icb_code', 'identifier', 'testing_period', 'cv_number',
                                                        'x_type','y_type'], keep='last')


    # use median for cross listing & multiple cross-validation
    detail_stock = detail_stock.groupby(['icb_code', 'identifier', 'testing_period', 'x_type', 'y_type',
                                         'label']).median()['pred'].reset_index(drop=False)

    detail_stock['icb_code'] = detail_stock['icb_code'].astype(float)  # convert icb_code to int
    yoy_med['icb_code'] = yoy_med['icb_code'].astype(float)

    # merge (stock prediction) with (ibes consensus median)
    yoy_merge = detail_stock.merge(yoy_med, left_on=['identifier', 'testing_period', 'y_type', 'icb_code','label'],
                                        right_on=['identifier', 'period_end', 'y_type', 'icb_code','label'],
                                        suffixes=('_lgbm', '_ibes'))

    # return label_sector(yoy_merge[['identifier', 'testing_period', 'y_type', 'x_type', 'pred', 'icb_code',
    #                                'y_consensus_qcut', 'y_ni_qcut', 'y_ibes_qcut', 'y_ibes', 'y_consensus']])
    return label_sector(yoy_merge)


if __name__ == "__main__":

    r_name_list = ['all x 0 -fix space', 'new with indi code -fix space',
                    'compare large space']

    r_name = 'small_space -code 0 -exclude_fwd True'

    r_name = 'mini_tune15 -code 0 -exclude_fwd True'
    r_name = 'new with indi code -fix space'
    r_name = 'mini_tune15_re -code 0 -exclude_fwd True'
    r_name = 'top15_infwd_mini -code 0 -exclude_fwd False'

    # r_name = 'new with indi code -fix space'
    # r_name = 'compare large space'
    # r_name = 'new industry model -fix space'
    # r_name = 'all x 0 -fix space'
    r_name = 'compare large space'

    # r_name = 'sp_fix_space -best_col 0 -code 0 -exclude_fwd True'

    tname = 'dense2'

    # for i in r_name:
    yoy_merge = merge_ibes_stock(sp_only=False)
    calc_mae_write(yoy_merge, r_name, tname='{}｜{}'.format(tname, r_name), base_list_type='all')

    combine()