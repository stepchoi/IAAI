from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import xarray
from sqlalchemy import create_engine

import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import gc

from miscel import check_dup, date_type
from preprocess.ratios import worldscope, full_period, trim_outlier

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

idd = 'C156E0340'
def check_id(df, id=idd):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.loc[df['identifier'] ==id, ['period_end', 'y_ibes']].sort_values(['period_end']))
    exit(0)

def read_data():

    ''' read worldscope_quarter_summary / ibes_data / stock_data / macro_data / clean_ratios'''

    try:  # read Worldscope Data after cleansing
        ws = pd.read_csv('preprocess/quarter_summary_clean.csv')    # raw worldscope data (i.e. point-in-time)
        ibes = pd.read_csv('preprocess/ibes_data.csv')              # raw ibes data
        y = pd.read_csv('preprocess/clean_ratios.csv', usecols=['identifier','period_end','y_ibes','y_ni'])     # Y ratios from clean table
        print('local version run - quarter_summary_clean / ibes_data / stock_data / macro_data / clean_ratios')
    except:
        print('---------------------> load rnn data')
        ws = worldscope().fill_missing_ws() # from Proprocess.ratios.py genenrate raw worldscope data
        with engine.connect() as conn:
            ibes = pd.read_sql('SELECT * FROM ibes_data', conn)     # use DB TABLE if no local file
            y = pd.read_sql('SELECT identifier, period_end, y_ibes, y_ni FROM clean_ratios', conn)
        engine.dispose()

    main = pd.merge(date_type(ws), date_type(ibes), on=['identifier','period_end'], how='left')  # convert ws to yoy
    main = main.merge(date_type(y), on=['identifier','period_end'], how='left')
    main.columns = [x.lower() for x in main.columns]    # convert columns name to lower case
    print(main)

    main = full_period(main)  # fill in non sequential records
    main['eps_rnn'] = (main['eps1tr12'] - main['eps1tr12'].shift(4)) / (main['fn_8001'].shift(4)) * (main['fn_5192'].shift(4))

    return main[['identifier','period_end', 'eps_rnn','y_ibes']]

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self):
        ''' split train and testing set
                    -> return dictionary contain (x, y, y without qcut) & cut_bins'''

        self.main = read_data()     # all YoY ratios
        self.cut_bins = {}
        self.sector = pd.DataFrame()
        self.train = pd.DataFrame()

    def split_entire(self, add_ind_code=0):   # we always use entire samples for training
        ''' train on all sample, add_ind_code = True means adding industry_code(2) as x '''
        self.sector = self.main

    def split_train_test(self, testing_period, qcut_q=10):
        ''' split training / testing set based on testing period '''

        # 1. split and qcut train / test Y
        start_train_y = testing_period - relativedelta(years=10)    # train df = 40 quarters
        self.sector = full_period(self.sector).sort_values(['period_end', 'identifier']).reset_index(drop=True)  # fill in for non-sequential records

        # 2.1. slice data for sample period + lookback period
        start_train = testing_period - relativedelta(years=15)    # train df = 10y + 5y lookback

        train_2dx_info = self.sector.loc[(start_train <= self.sector['period_end']) & (self.sector['period_end'] <= testing_period)] # extract df for X

        # 2.2. standardize data
        train_2dx_info = train_2dx_info.groupby(['identifier','period_end'])['eps_rnn'].median().unstack()

        return train_2dx_info

    def standardize_x(self, train_x, test_x):
        ''' tandardize x with train_x fit '''

        scaler = StandardScaler().fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        return train_x, test_x

    def y_qcut(self, train_y, test_y, qcut_q):
        ''' qcut y '''

        ''' convert qcut bins to median of each group '''

        self.cut_bins = {}

        # cut original series into 0, 1, .... (bins * n)
        train_y_qcut, self.cut_bins['cut_bins'] = pd.qcut(train_y, q=qcut_q, retbins=True, labels=False)
        test_y_qcut = pd.cut(test_y, bins=self.cut_bins['cut_bins'], labels=False)

        # calculate median on train_y for each qcut group
        df = pd.DataFrame(np.vstack((train_y, np.array(train_y_qcut)))).T   # concat original series / qcut series
        self.cut_bins['med_train'] = df.groupby([1]).median().sort_index()[0].to_list()     # find median of each group

        # replace 0, 1, ... into median
        train_y = pd.DataFrame(train_y_qcut).replace(range(qcut_q), self.cut_bins['med_train']).values
        test_y = pd.DataFrame(test_y_qcut).replace(range(qcut_q), self.cut_bins['med_train']).values

        return train_y, test_y


if __name__ == '__main__':

    add_ind_code = 0
    testing_period = dt.datetime(2013, 6, 30)
    qcut_q = 10
    exclude_fwd = False
    small_training = True
    eps_only = True

    data = load_data()
    data.split_entire(add_ind_code)
    train_x, train_y, X_test, Y_test, cv, test_id, x_col = data.split_train_test(testing_period)

    print(x_col)
    print(train_x.shape, X_test.shape)
    exit(0)

    for train_index, test_index in cv:
        X_train = train_x[train_index]
        Y_train = train_y[train_index]
        X_valid = train_x[test_index]
        Y_valid = train_y[test_index]

        print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

