from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt

from miscel import check_dup, date_type
from preprocess.ratios import worldscope

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

'''
1. read DB TABLE
2. fill 0
3. pivot 3D array
'''

class add_macro:

    def __init__(self, ratios, macros):
        self.ratios = self.label_nation_sector(ratios)
        self.macros = macros

    def label_nation_sector(self, ratios):
        ''' read files mapping icb_code & market '''

        with engine.connect() as conn:
            markets = pd.read_sql("SELECT index_ric, icb_sector, identifier FROM dl_value_universe "
                                  "WHERE identifier IS NOT NULL", conn)
        engine.dispose()

        ratios = pd.merge(ratios, markets, on=['identifier'])
        ratios['market'] = ratios['index_ric'].replace(['0#.CSI300', '0#.N225', '0#.SPX', '0#.HSLI'], ['CH','JP','US','HK'])
        return ratios.drop(['index_ric'], axis=1)

    def map_macros(self):
        ''' map macros to ratios TABLE '''

        with engine.connect():
            mapping = pd.read_sql('SELECT * FROM macro_formula', engine).iloc[:,:3]
        engine.dispose()

        # map worldwide indicators based on period_end
        worldwide_col = mapping.loc[mapping['market'] == 'W', 'symbol'].to_list()
        self.ratios = pd.merge(self.ratios, self.macros[['period_end'] + worldwide_col], on=['period_end'], how='left')

        # map market-specific indicators based on period_end & market
        new_macros = self.macros.set_index(['period_end']).unstack().reset_index() # pivot -> index (period_end, market)
        new_macros.columns = ['symbol', 'period_end', 'values']
        new_macros = pd.merge(new_macros, mapping, on=['symbol'])
        new_macros = new_macros.loc[new_macros['market'] != 'W']

        new_macros = new_macros.pivot_table(index=['period_end','market'], columns='type', values='values')
        self.ratios = pd.merge(self.ratios, new_macros, on=['period_end', 'market'], how='left')

        self.ratios['period_end'] = pd.to_datetime(self.ratios['period_end'])
        return self.ratios

def read_data():

    ''' read worldscope_quarter_summary / ibes_data / stock_data '''

    try:  # read Worldscope Data after cleansing
        ws = pd.read_csv('preprocess/quarter_summary_clean.csv')
        ibes = pd.read_csv('preprocess/ibes_data.csv')
        stock = pd.read_csv('preprocess/stock_data.csv')
        macro = pd.read_csv('preprocess/macro_data.csv')
        y = pd.read_csv('preprocess/clean_ratios.csv', usecols=['identifier','period_end','y_ibes','y_ni'])
        print('local version run - quarter_summary_clean / ibes_data / stock_data ')
    except:
        ws = worldscope().fill_missing_ws()
        with engine.connect() as conn:
            ibes = pd.read_sql('SELECT * FROM ibes_data', conn)
            stock = pd.read_sql('SELECT * FROM stock_data', conn)
            macro = pd.read_sql('SELECT * FROM macro_data', conn)
            y = pd.read_sql('SELECT identifier, period_end, y_ibes, y_ni FROM clean_ratio', conn)
        engine.dispose()

    main = pd.merge(ws, ibes, on=['identifier','period_end'])
    main = main.merge(stock, on=['identifier','period_end'])
    main = add_macro(main, macro).map_macros()
    main = main.merge(y, on=['identifier','period_end'])
    print(main.shape, main.columns)

    check_dup(main)

if __name__ == '__main__':
    read_data()
