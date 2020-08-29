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
        detail_stock = pd.read_csv('results_analysis/compare_with_ibes/{}_stock_{}.csv'.format(tname, r_name))
        detail_stock = date_type(detail_stock, date_col='testing_period')
        print('local version run - stock_{}'.format(r_name))
    except:
        print('----------------> update stock results from DB TABLE {}'.format(tname))

        with engine.connect() as conn:
            # read DB TABLE results_lightgbm data for given "name"
            if r_name == 'all':
                if tname == 'cnn_rnn':
                    result_all = pd.read_sql(
                        "SELECT * FROM results_{} WHERE trial_lgbm > 181".format(tname, r_name), conn)       # change for results_rnn_eps
                else:
                    result_all = pd.read_sql(
                        "SELECT * FROM results_{}".format(tname, r_name), conn)  # change for results_rnn_eps
            else:
                result_all = pd.read_sql(
                    "SELECT * FROM results_{} WHERE name='{}'".format(tname, r_name), conn)

            trial_lgbm = set(result_all['trial_lgbm'])

            # read corresponding part of DB TABLE results_lightgbm_stock
            query = text('SELECT * FROM results_{}_stock WHERE (trial_lgbm IN :trial_lgbm)'.format(tname))
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_stock = pd.read_sql(query, conn)
        engine.dispose()

        stock_label = ['trial_lgbm', 'icb_code', 'testing_period', 'cv_number']
        detail_stock = result_stock.merge(result_all, on=stock_label, how='inner')  # map training information to stock data
        detail_stock.to_csv('results_analysis/compare_with_ibes/{}_stock_{}.csv'.format(tname, r_name), index=False)
        print('detial_stock shape: ', detail_stock)

    print(detail_stock)

    return detail_stock

def merge_ibes_stock():
    ''' merge ibes and detail stock data '''

    yoy_med = download_ibes_median()
    detail_stock = download_stock()

    detail_stock['y_type'] = 'ibes'     # all rnn trials has been ibes_yoy as Y
    detail_stock['label'] = 'rnn'       # "label" to find cut_bins from TABLE results_bins_new

    # if tname == 'rnn_top':
    # detail_stock['exclude_fwd'] = False

    detail_stock = detail_stock.loc[detail_stock['icb_code']==0]
    detail_stock = detail_stock.loc[detail_stock['exclude_fwd']==True]

    # decide base list -> identifier + period_end appeared in both lgbm and rnn models
    lgbm = pd.read_csv('results_analysis/compare_with_ibes/stock_ibes_new industry_only ws -indi space3.csv',
                       usecols=['identifier', 'testing_period'])
    rnn = pd.read_csv('results_analysis/compare_with_ibes/rnn_eps_stock_all.csv',
                      usecols=['identifier', 'testing_period'])
    base_list = pd.merge(lgbm, rnn, on=['identifier', 'testing_period'], how='inner')
    base_list = date_type(base_list, 'testing_period')

    detail_stock = detail_stock.merge(base_list, on=['identifier', 'testing_period'], how='right')

    if tname == 'rnn_eps':
        detail_stock['x_type'] = 'fwdepsqcut'
    else:
        detail_stock['exclude_fwd'] = detail_stock['exclude_fwd'].fillna(False)     # exclude_fwd default is False
        x_type_dic = {False: 'ni', True: 'fwdepsqcut'}  # False means all_x (include ibes); True means no ibes data
        detail_stock['x_type'] = [x_type_dic[x] for x in detail_stock['exclude_fwd']]   # convert to x_type name

    detail_stock = detail_stock.drop_duplicates(subset=['icb_code', 'identifier', 'testing_period', 'cv_number','y_type'], keep='last')

    if 'industry' not in r_name:    # for aggregate model use 0/1/2 to represent different x (in fact same samples)
        print('------ convert entire ------')
        detail_stock.loc[detail_stock['icb_code'] == 1, 'x_type'] += '-industry_code'   # 1 means include industry_code_x
        detail_stock.loc[detail_stock['icb_code'] == 2, 'x_type'] += '-sector_code'     # 2 means include sector_code_x
        detail_stock['icb_code'] = 0

    # use median for cross listing & multiple cross-validation
    detail_stock = detail_stock.groupby(['icb_code','identifier','testing_period','x_type','y_type','label']).median()['pred'].reset_index(drop=False)

    detail_stock['icb_code'] = detail_stock['icb_code'].astype(float)  # convert icb_code to int
    yoy_med['icb_code'] = yoy_med['icb_code'].astype(float)

    # merge (stock prediction) with (ibes consensus median)
    yoy_merge = detail_stock.merge(yoy_med, left_on=['identifier', 'testing_period', 'y_type', 'icb_code', 'label'],
                                        right_on=['identifier', 'period_end', 'y_type', 'icb_code', 'label'],
                                        suffixes=('_lgbm', '_ibes'))
    return label_sector(yoy_merge)

if __name__ == "__main__":

    # organize()

    # r_name = 'small_training_False_0'
    # r_name = 'without ibes -2'
    r_name = 'industry_exclude'
    # r_name = 'new_without_ibes'
    # r_name = 'top15'
    tname = 'cnn_rnn' # or rnn_eps
    #
    # r_name = 'top15_lgbm'
    # tname = 'rnn_top'



    yoy_merge = merge_ibes_stock()
    calc_mae_write(yoy_merge, r_name, tname='{}｜{}'.format(tname, r_name))

    combine()