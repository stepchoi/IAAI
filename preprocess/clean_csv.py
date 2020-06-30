import numpy as np
import pandas as pd
import datetime as dt
from sqlalchemy import create_engine, text

def clean_ibes():

    ''' organize multiple excel sheet for IBES data to single TABLE '''

    ibes_hk = pd.read_excel('preprocess/IBES_data_raw.xlsx', 'HK')
    ibes_us = pd.read_excel('preprocess/IBES_data_raw.xlsx', 'US')

    def clean(df):

        # remove unused columns
        ticker_symbol = pd.DataFrame(index=df['symbol'].dropna().to_list(),
                          columns=df['ticker'].dropna().to_list()).unstack().reset_index().iloc[:,:2]
        ticker_symbol.columns = ['ticker', 'symbol']
        new_df = pd.concat([ticker_symbol, df.iloc[:,3:].dropna(how='all')], axis=1)
        new_df = new_df.loc[new_df['symbol'].isin(['EBD1FD12', 'CAP1FD12', 'EPS1FD12'])].drop_duplicates(subset=['ticker', 'symbol'])

        # map ticker to identifier
        id = pd.read_sql('SELECT member_ric, ws_identifier FROM dss_index_members WHERE ws_identifier IS NOT NULL', engine)
        id.columns = ['ticker', 'identifier']
        new_df = pd.merge(new_df, id, on = ['ticker']).set_index(['symbol', 'identifier']).iloc[:, 1:].unstack().T.reset_index()

        # covert quarter date -> period end
        new_df[['quarter','year']] = new_df['level_0'].str.split(' ',expand=True)
        new_df['quarter'] = new_df['quarter'].replace(['Q1','Q2','Q3','Q4'], ['0331', '0630', '0930', '1231'])
        new_df['period_end'] = new_df['year'] + new_df['quarter']
        new_df['period_end'] = pd.to_datetime(new_df['period_end'], format='%Y%m%d')
        new_df = new_df.filter(['identifier', 'period_end', 'EBD1FD12', 'CAP1FD12', 'EPS1FD12'])

        return new_df

    clean_ibes_us = clean(ibes_us)
    clean_ibes = pd.concat([clean_ibes_hk, clean_ibes_us], axis = 0)

    for col in ['EBD1FD12', 'CAP1FD12', 'EPS1FD12']:
        clean_ibes[col] = pd.to_numeric(clean_ibes[col], errors='coerce') # convert ERROR STRING to NaN

    clean_ibes = clean_ibes.dropna(subset=['EBD1FD12', 'CAP1FD12', 'EPS1FD12'], how='all')
    print(clean_ibes)

    clean_ibes.to_csv('ibes_data.csv', index=False)
    # clean_ibes.to_sql('ibes_data', engine, if_exists='replace', index=False)
    print(clean_ibes)

def clean_dss_index_members():
    ''' arrrange TABLE dss_index_members:   1. rename ws_identifier -> identifier
                                            2. add icb_sector '''

    df = pd.read_sql('SELECT * FROM dss_index_members', engine)
    # df.to_csv('clean_dss_index_members.csv', index=False)

    sectors = pd.read_sql("SELECT identifier, left(data, 6) as icb_code FROM worldscope_static "
                          "WHERE field_number = '7040.0'", engine)
    df = pd.merge(df, sectors, left_on=['ws_identifier'], right_on=['identifier'], how='left')
    df = df.filter(['index_ric', 'member_ric', 'year', 'month', 'definition_date', 'ws_identifier', 'icb_code'])
    df.columns = ['index_ric', 'member_ric', 'year', 'month', 'definition_date', 'identifier', 'icb_sector']

    df.to_csv('dl_value_universe.csv', index=False)
    print('saved')
    df.to_sql('dl_value_universe', engine, if_exists='replace')


def check_ws_available():
    df = pd.read_sql('select distinct field_number from worldscope_quarter order by field_number asc;', engine)['field_number'].to_list()
    print(df)

if __name__ == '__main__':
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    check_ws_available()
