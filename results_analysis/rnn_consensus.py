from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
from preprocess.ratios import full_period, worldscope
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
        detail_stock = pd.read_csv('results_analysis/compare_with_ibes/rnn_stock_{}.csv'.format(r_name))
        detail_stock = date_type(detail_stock, date_col='testing_period')
        print('local version run - stock_{}'.format(r_name))
    except:
        print('----------------> update stock results from DB TABLE {}'.format(tname))

        with engine.connect() as conn:
            # read DB TABLE results_lightgbm data for given "name"
            if r_name == 'all':
                result_all = pd.read_sql(
                    "SELECT trial_lgbm, icb_code, testing_period, cv_number, exclude_fwd "
                    "FROM results_{} WHERE trial_lgbm > 181".format(tname, r_name), conn)       # change for results_rnn_eps
            else:
                result_all = pd.read_sql(
                    "SELECT trial_lgbm, icb_code, testing_period, cv_number, exclude_fwd "
                    "FROM results_{} WHERE name='{}'".format(tname, r_name), conn)

            trial_lgbm = set(result_all['trial_lgbm'])

            # read corresponding part of DB TABLE results_lightgbm_stock
            query = text('SELECT * FROM results_{}_stock WHERE (trial_lgbm IN :trial_lgbm)'.format(tname))
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_stock = pd.read_sql(query, conn)
        engine.dispose()

        stock_label = ['trial_lgbm', 'icb_code', 'testing_period', 'cv_number']
        detail_stock = result_stock.merge(result_all, on=stock_label, how='inner')  # map training information to stock data
        detail_stock.to_csv('results_analysis/compare_with_ibes/rnn_stock_{}.csv'.format(r_name), index=False)
        print('detial_stock shape: ', detail_stock)

    print(detail_stock)

    return detail_stock

def merge_ibes_stock():
    ''' merge ibes and detail stock data '''

    yoy_med = download_ibes_median()
    detail_stock = download_stock()

    detail_stock['x_type'] = 'fwdepsqcut'
    detail_stock['y_type'] = 'ibes'

    detail_stock = detail_stock.drop_duplicates(subset=['icb_code', 'identifier', 'testing_period', 'cv_number',
                                                        'y_type'], keep='last')

    print('------ convert entire ------')
    detail_stock.loc[detail_stock['icb_code'] == 1, 'x_type'] = 'fwdepsqcut-industry_code'
    detail_stock.loc[detail_stock['icb_code'] == 2, 'x_type'] = 'fwdepsqcut-sector_code'
    detail_stock['icb_code'] = 0

    # use median for cross listing & multiple cross-validation
    detail_stock = detail_stock.groupby(['icb_code','identifier','testing_period','x_type','y_type']).median()[
        'pred'].reset_index(drop=False)

    detail_stock['icb_code'] = detail_stock['icb_code'].astype(float)  # convert icb_code to int
    yoy_med['icb_code'] = yoy_med['icb_code'].astype(float)

    # merge (stock prediction) with (ibes consensus median)
    yoy_merge = detail_stock.merge(yoy_med, left_on=['identifier', 'testing_period', 'y_type', 'icb_code'],
                                        right_on=['identifier', 'period_end', 'y_type', 'icb_code'],
                                        suffixes=('_lgbm', '_ibes'))

    return label_sector(yoy_merge[['identifier', 'testing_period', 'y_type', 'x_type', 'pred', 'icb_code',
                                   'y_consensus_qcut', 'y_ni_qcut', 'y_ibes_qcut', 'y_ibes', 'y_consensus']])

def organize():
    ''' match records in results_cnn_rnn and results_cnn_rnn_stock '''

    stock = pd.read_csv('#cnn_rnn_stock.csv')
    info = pd.read_csv('#cnn_rnn.csv')

    info = info.loc[~info['name'].isin(['without ibes', 'small_training_True_0'])]
    info = info.loc[info['trial_lgbm']>181]
    print(stock.columns)

    # def rename_dup(x):   # rename same trial_lgbm to different by time
    #     counter = ['{}_'.format(e) for e in x.groupby(x).cumcount().add(1).astype(str)]
    #     return x.mask(x.duplicated(), x.astype(str).radd(counter))
    #
    # info['trial_lgbm_unique'] = rename_dup(info['trial_lgbm'])

    # i_set = []
    # for i, g in stock.groupby([(stock['trial_lgbm'] != stock['trial_lgbm'].shift()).cumsum()]):
    #     i_set.append(g['trial_lgbm'].tolist()[0])
    #
    # print(i_set)
    # exit(1)

    stock_label = ['trial_lgbm', 'icb_code', 'testing_period', 'cv_number']
    df = pd.merge(stock, info, on=stock_label, how='left')
    print(df)

    dup = df.loc[df.duplicated(stock_label + ['identifier'], keep=False)]

    # df.to_csv('stock_rnn.csv', index=False)


if __name__ == "__main__":

    organize()

    r_name = 'all'
    tname = 'cnn_rnn' # or rnn_eps

    yoy_merge = merge_ibes_stock()
    print(yoy_merge)

    calc_mae_write(yoy_merge, tname='{}｜{}'.format(tname, r_name))

    combine()