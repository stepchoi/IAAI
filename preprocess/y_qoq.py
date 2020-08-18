from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt
from preprocess.ratios import full_period, worldscope

from miscel import check_dup, date_type

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def clean_ibes_excel():
    ''' clean ibes quarter data from excel -> csv'''

    with engine.connect() as conn:  # read ticker -> id mapping
        id = pd.read_sql('SELECT member_ric, identifier FROM dl_value_universe WHERE identifier IS NOT NULL',
                         conn).set_index('member_ric')
    engine.dispose()

    df_list = []
    for sheet in ['CH','JP','US','HK']:
        raw = pd.read_excel('preprocess/Downloaded IBES_quarter.xlsx', sheet, header=None) # read_excel
        tic = raw[0].dropna().to_list()  # find ticker list (col 0)
        raw = raw.iloc[:,2:].dropna(subset=[2]).reset_index(drop=True)  # remove ticker/col columns from data
        idx = raw.loc[raw[2]=='Name', 2].index[1] # find the starting row for 10EPS

        # seperate selection for consensus/actual
        c = raw.iloc[1:idx,:].set_index(2)
        a = raw.iloc[(1+idx):,:].set_index(2)

        def clean_df(df, col_name): # pivot table
            df.columns = tic # rename column as ticker
            df = df.unstack().reset_index()
            df.columns = ['member_ric','period_end', col_name]
            df = df.merge(id, on='member_ric')
            return df[['identifier','period_end',col_name]]

        c = clean_df(c, 'epsi1md')  # reorganize consensus
        a = clean_df(a, 'i0eps')    # reorganize actual
        new_df = pd.merge(c,a, on=['identifier','period_end'], how='outer') # combine two columns (consensus + actual)
        print(sheet, new_df.shape)
        df_list.append(new_df)

    pd.concat(df_list, axis=0).to_csv('preprocess/ibes_data_quarter.csv', index=False)

def clean_ibes_excel2():
    ''' convert all field to Numberic + period_end (replace old ibes_quarter file) '''

    df = pd.read_csv('preprocess/ibes_data_quarter.csv')
    df[['epsi1md', 'i0eps']] = df[['epsi1md', 'i0eps']].apply(pd.to_numeric, errors='coerce')   # all to numeric
    
    # covert quarter date -> period end
    df[['quarter', 'year']] = df['period_end'].str.split(' ', expand=True)
    df['quarter'] = df['quarter'].replace(['Q1', 'Q2', 'Q3', 'Q4'], ['0331', '0630', '0930', '1231'])
    df['period_end'] = df['year'] + df['quarter']
    df['period_end'] = pd.to_datetime(df['period_end'], format='%Y%m%d')

    df.drop(['quarter', 'year'], axis=1).to_csv('preprocess/ibes_data_quarter.csv', index=False)

def eps_to_qoq():
    ''' convert quarter to qoq '''

    ibes = pd.read_csv('preprocess/ibes_data_quarter.csv')
    ibes = ibes.groupby(['identifier', 'period_end']).mean().reset_index(drop=False)  # for cross listing use average

    try:  # read Worldscope Data after cleansing
        ws = pd.read_csv('preprocess/quarter_summary_clean.csv',
                              usecols=['identifier', 'period_end', 'fn_18263', 'fn_8001', 'fn_5192'])
        print('local version run - quarter_summary_clean (with period_end) ')
    except:
        ws = worldscope().fill_missing_ws()[['identifier', 'period_end', 'fn_18263', 'fn_8001', 'fn_5192']]

    ibes = ibes.merge(ws, on=['identifier', 'period_end'])  # combines ibes + worldscope (for mcap + share)

    ibes = full_period(date_type(ibes)) # using full period to calculate
    ibes['y_ibes_qoq'] = (ibes['i0eps'].shift(-1) - ibes['i0eps']) / ibes['fn_8001'] * ibes['fn_5192']
    ibes.loc[ibes.groupby('identifier').tail(1).index, 'y_ibes_qoq'] = np.nan  # use ibes ttm for Y

    ibes['y_consensus_qoq'] = (ibes['epsi1md'] - ibes['i0eps']) * ibes['fn_5192'] / ibes['fn_8001']  # use ibes fwd & ttm for Y estimation

    ibes.dropna(subset=ibes.columns[2:], how='all')[['identifier', 'period_end','y_ibes_qoq','y_consensus_qoq']].to_csv('preprocess/ibes_data_qoq.csv', index=False)

if __name__ == '__main__':
    # clean_ibes_excel()
    # clean_ibes_excel2()
    eps_to_qoq()
