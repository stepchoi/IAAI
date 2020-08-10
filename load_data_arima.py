from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import datetime as dt
from dateutil.relativedelta import relativedelta

from miscel import check_dup, date_type
from preprocess.ratios import worldscope, full_period, trim_outlier

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

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

    def split_train_test(self, testing_period):
        ''' split training / testing set based on testing period '''

        # 1. split and qcut train / test Y
        self.main = full_period(self.main).sort_values(['period_end','identifier']).reset_index(drop=True)  # fill in for non-sequential records

        # 2.1. slice data for sample period + lookback period
        start_train = testing_period - relativedelta(years=15)    # train df = 10y + 5y lookback
        end_train = testing_period + relativedelta(years=1)    # train df = 10y + 5y lookback
        train_2dx_info = self.main.loc[(start_train < self.main['period_end']) &
                                         (self.main['period_end'] <= end_train)] # extract df for X
        # 2.2. standardize data
        train_2dx_info = train_2dx_info.groupby(['identifier','period_end'])['eps_rnn'].median().unstack()

        train_2dx_info.iloc[:,:] = convert_to_median(train_2dx_info, testing_period) # convert x/y to median based on lgbm records

        return train_2dx_info

def convert_to_median(df, testing_period):
    '''  convert Y in qcut format to medians with med_train from training set'''

    with engine.connect() as conn:
        bins_df = pd.read_sql('SELECT * from results_bins_new WHERE icb_code=1', conn)
    engine.dispose()

    cut_bins_dict = bins_df.loc[bins_df['testing_period'] == testing_period].iloc[0,:].to_dict()

    cut_bins = cut_bins_dict['cut_bins'].strip('[]').split(',')   # convert string {1, 2, 3....} to list
    med_test = cut_bins_dict['med_train'].strip('[]').split(',')

    cut_bins[0] = -np.inf  # convert cut_bins into [-inf, ... , inf]
    cut_bins[-1] = np.inf
    cut_bins = [float(x) for x in cut_bins]     # convert string in list to float
    med_test = [float(x) for x in med_test]

    arr_q = df.apply(pd.cut, bins=cut_bins, labels=False)  # cut original series into 0, 1, .... (bins * n)
    arr_q = arr_q.replace(range(int(cut_bins_dict['qcut_q'])), med_test).values  # replace 0, 1, ... into median

    return arr_q  # return converted Y and median of all groups

if __name__ == '__main__':

    testing_period = dt.datetime(2013, 6, 30)
    exclude_fwd = False
    small_training = True
    eps_only = True

    data = load_data()
    train_x = data.split_train_test(testing_period)

    print(train_x.shape)


