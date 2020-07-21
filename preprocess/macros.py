from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import datetime as dt
from preprocess.ratios import trim_outlier
from miscel import date_type

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def eikon_to_datetime(df, date_col):
    ''' convert quarter time e.g. Q2 1998 -> 1998-03-31 datetime'''
    df[['quarter', 'year']] = df[date_col].str.split(' ', expand=True)
    df['quarter'] = df['quarter'].replace(['Q1', 'Q2', 'Q3', 'Q4'], ['0331', '0630', '0930', '1231'])
    df['period_end'] = df['year'] + df['quarter']
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y%m%d')
    df['year'] = df['year'].astype(int)
    return df

def reer_or_neer(macro):
    ''' calculate one-way ANOVA whether Q1 - Q4 have different average for 1996 - 2020 period'''

    neer_col = ['NOMINAL EER - BROAD - TRADE WEIGHTED', 'NOMINAL EER - BROAD - TRADE WEIGHTED.1', 'NOMINAL EER - BROAD - TRADE WEIGHTED.2']
    macro = macro.dropna(subset = neer_col, how='any')

    import scipy.stats as stats

    for i in neer_col:
        result = stats.f_oneway(    macro[i][macro['quarter'] == '0331'],
                                    macro[i][macro['quarter'] == '0630'],
                                    macro[i][macro['quarter'] == '0930'],
                                    macro[i][macro['quarter'] == '1231'])
        print(i, result)

    # result is not significant, so we use REER instead that includes more info
    # NOMINAL EER - BROAD - TRADE WEIGHTED F_onewayResult(statistic=0.11956050222671315, pvalue=0.948394333734049)
    # NOMINAL EER - BROAD - TRADE WEIGHTED.1 F_onewayResult(statistic=0.0686270035149266, pvalue=0.9764916778772617)
    # NOMINAL EER - BROAD - TRADE WEIGHTED.2 F_onewayResult(statistic=0.013839849498256096, pvalue=0.9977607688630413)

def clean_macro():
    ''' read all macro csv and combine tables -> upload to DB
        (need local file: macro_data_raw.xlsx) '''

    macros = pd.read_excel('macro_data_raw.xlsx', 'macros')
    macros = eikon_to_datetime(macros, 'Name') # convert quarter (Q1) -> period end (3-31)

    reer = pd.read_excel('macro_data_raw.xlsx', 'reer')
    index = pd.read_excel('macro_data_raw.xlsx', 'index')
    index = date_type(index) # convert string to datetime

    interest_rate = pd.read_excel('macro_data_raw.xlsx', 'interest rate')
    interest_rate['Name'] = pd.to_datetime(interest_rate['Name'], format='%Y-%m-%d') # convert string to datetime

    macros = pd.merge(macros, reer, on=['year'])
    macros = pd.merge(macros, index, left_on = ['period_end'], right_on=['Name'], how='left')
    macros = pd.merge(macros, interest_rate, left_on = ['period_end'], right_on=['Name'], how='left')

    num_col = macros.columns[macros.dtypes=='float64'].to_list()
    macros = macros[['period_end'] + num_col]

    macros.columns = [x.lower() for x in macros.columns]
    print(macros)

    macros.to_csv('macro_data.csv', index=False)

    return macros

class macro_calc:

    def __init__(self):
        ''' read raw data + mapping table '''

        try:
            self.macros = pd.read_csv('preprocess/macro_data.csv')
            mapping = pd.read_csv('preprocess/macro_formula.csv')
            print('local version run - macros, mapping')
        except:
            self.macros = pd.read_sql('SELECT * FROM macro_data', engine)
            mapping = pd.read_sql('SELECT * FROM macro_formula', engine)

        self.yoy_col = list(set(mapping.loc[mapping['yoy'], 'symbol'].to_list()))   # columns need yoy conversion
        self.push_1q_col = mapping.loc[mapping['1q_ahead'], 'symbol'].to_list()     # i.e. market variables

    def calc_yoy(self):
        ''' calculate yoy & push quarters for market variables '''

        macro_ratios = self.macros.drop(self.yoy_col, axis=1)
        macro_ratios['chgdp%..c'] = macro_ratios['chgdp%..c'].div(100) # convert china gdp (source EIKON) from %
        macro_ratios[self.yoy_col] = (self.macros[self.yoy_col] / self.macros[self.yoy_col].shift(4)).sub(1)

        macro_ratios = self.push_1q(macro_ratios)

        return date_type(macro_ratios)

    def push_1q(self, df):
        df[self.push_1q_col] = df[self.push_1q_col].shift(-1)
        return df

def new_macro_ratios():
    ''' new macro ratios '''

    macro = pd.read_excel('preprocess/New Macro Variable Values.xlsx','Sheet1')
    macro['month'] = macro['Name'].apply(lambda x: x.month)
    macro = macro.loc[macro['month'].isin([3,6,9,12])]
    next = {3:4, 6:7, 9:10, 12:1}
    macro['period_end'] = macro['Name'].apply(lambda x: pd.Timestamp(x.year, next[x.month], 1) - dt.timedelta(days=1))

    oil = pd.read_excel('preprocess/New Macro Variable Values.xlsx','Sheet2').sort_values('Name')
    oil['year_month'] = oil['Name'].apply(lambda x: '{}_{}'.format(x.year, x.month))
    oil_end = oil.groupby('year_month').last()                      # find month end record from daily records
    oil_end['month'] = oil_end['Name'].apply(lambda x: x.month)
    oil_end = oil_end.loc[oil_end['month'].isin([3,6,9,12])]        # use quarter end month
    oil_end['period_end'] = oil_end['Name'].apply(lambda x: pd.Timestamp(x.year, next[x.month], 1) - dt.timedelta(days=1))

    new_macro = pd.merge(date_type(macro.drop(['Name','month'], axis=1)),
                         oil_end[['period_end','CRUDOIL']], on='period_end').sort_values(['period_end'])
    new_macro.columns = [x.lower() for x in new_macro.columns.to_list()]
    new_macro.to_csv('preprocess/macro_data_new.csv', index=False)

    yoy_col = list(set(new_macro.columns.to_list()) - {'period_end'})
    new_macro[yoy_col] = (new_macro[yoy_col] / new_macro[yoy_col].shift(4)).sub(1)
    new_macro.to_csv('preprocess/clean_macros_new.csv', index=False)

    return new_macro


if __name__ == '__main__':
    new_macro_ratios()

    macro_ratios =  macro_calc().calc_yoy()

    # replace -np.inf with minimum value in usfrbpim
    real_min = macro_ratios['usfrbpim'].replace([-np.inf, np.inf], np.nan).min()
    real_max = macro_ratios['usfrbpim'].replace([-np.inf, np.inf], np.nan).max()
    macro_ratios['usfrbpim'] = macro_ratios['usfrbpim'].replace([-np.inf, np.inf], [real_min, real_max])
    print(macro_ratios.describe().T[['max','min']])

    macro_ratios.to_csv('preprocess/clean_macros.csv', index=False)

    # worldwide_col = mapping.loc[mapping['market'] == 'W', 'symbol'].to_list()