from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt

from miscel import check_dup, date_type
from preprocess.x_ratios import worldscope, full_period, trim_outlier

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

class calc_ts_qoq:

    def __init__(self, ws):
        ''' add full period for ws '''
        self.ts_dict = {'fn_18263': 'ni', 'fn_18262': 'sales', 'fn_18304': 'pretax_margin', 'fn_18265': 'cfps',
                        'EPS1TR12': 'eps'}
        self.ts_col = list(self.ts_dict.keys())
        self.ws = full_period(date_type(ws[['identifier', 'period_end'] + self.ts_col]), 'identifier')  # add missing records

    def calc_x(self):
        ''' calculate qoq time-series ratios for 4Q in the last year '''

        ts_ratio = self.ws[['identifier', 'period_end']]

        def ts_name(name_str): # raname ts ratios e.g. ni_ts01 for y-1 ~ y0
            return [x + '_' + name_str for x in list(self.ts_dict.values())]

        ts_ratio[ts_name('qoq0')] = (self.ws[self.ts_col]/self.ws[self.ts_col].shift(1)).sub(1)
        ts_ratio.loc[self.ws.groupby('identifier').head(1).index, ts_name('qoq0')] = np.nan # last quarter qoq

        final_df = ts_ratio.dropna(subset=ts_ratio.columns[2:].to_list(), how='all')
        final_df[final_df.columns[2:].to_list()] = trim_outlier(final_df[final_df.columns[2:].to_list()])
        # print(final_df.describe().T[['max', 'min']])
        final_df.to_csv('preprocess/ratios_qoq.csv', index=False)

        with engine.connect() as conn:
            final_df.to_sql('ratios_qoq', con=conn, if_exists='replace', method='multi', index=False)
        engine.dispose()

        return ts_ratio

if __name__ == '__main__':
    try:  # read Worldscope Data after cleansing
        ws = pd.read_csv('preprocess/quarter_summary_clean.csv')  # raw worldscope data (i.e. point-in-time)
        ibes = pd.read_csv('preprocess/ibes_data.csv')
    except:
        print('---------------------> load clean ws data')
        ws = worldscope().fill_missing_ws()  # from Proprocess.x_ratios.py genenrate raw worldscope data
        with engine.connect() as conn:
            ibes = pd.read_sql('SELECT * FROM ibes_data', conn) # ibes_data is the cleaned with prep_clean_csv.py and uploaded
        engine.dispose()

    ibes = ibes.groupby(['identifier', 'period_end']).mean().reset_index(drop=False)  # for cross listing use average
    ibes['identifier'] = ibes['identifier'].apply(lambda x: x.zfill(9)) # zfill identifiers with leading 0
    ws = pd.merge(ibes, ws, on=['identifier','period_end'])

    calc_ts_qoq(ws).calc_x()