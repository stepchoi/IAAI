from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import datetime as dt

'''
1. convert ibes eps back to net income
    a. find fn_5191 of each Q
2. convert to qcut + median <- use TABLE results_bins

3. merge with our result
4. calculate comparable accuracy <- for available stock used in our prediction

'''

def eps_to_ni_yoy():
    ''' calculate 1. IBES forward NET INCOME =
                     IBES forward EPS * common share outstanding used to calculate EPS (both at T0)
                  2. YoY = (forward NI - actual current NI) / current Market Cap
    '''

    # read original for calculation
    try:
        ibes = pd.read_csv('preprocess/ibes_data.csv', usecols = ['identifier', 'period_end','EPS1FD12'])
        ws = pd.read_csv('preprocess/ws_ibes.csv')
        print('local version run - ibes / share_osd')
    except:
        with engine.connect() as conn:
            ibes = pd.read_sql('SELECT identifier, period_end, eps1fd12 FROM ibes_data', conn)
            ws = pd.read_sql('SELECT identifier, year, frequency_number, fn_18263, fn_8001, fn_5192 as share_osd '
                                    'FROM worldscope_quarter_summary', conn)
        engine.dispose()

    ws.to_csv('preprocess/ws_ibes.csv', index=False)
    ibes['period_end'] = pd.to_datetime(ibes['period_end'], format='%Y-%m-%d')
    ibes.columns = ['identifier', 'period_end', 'eps1fd12']

    # map common share outstanding & market cap to ibes estimations
    ws = label_period_end(ws)
    ibes = ibes.merge(ws, on=['identifier', 'period_end'])

    # calculate YoY (Y)
    ibes['fwd_ni'] = ibes['eps1fd12'] * ibes['share_osd']
    ibes['y_ibes'] = (ibes['fwd_ni'] - ibes['fn_18263']) / ibes['fn_8001']

    # print(ibes)

    return ibes[['identifier', 'period_end', y_ibes]]

def label_period_end(df):
    ''' find fiscal_period_end -> last_year_end for each identifier + frequency_number * 3m '''

    try:
        fiscal_year_end = pd.read_csv('preprocess/useless/static_fiscal_year_end.csv')
    except:
        fiscal_year_end = pd.read_sql("SELECT identifier, data as fiscal_year_end FROM worldscope_static"
                                      " WHERE field_number = '5352.0'", engine)

    df = pd.merge(df, fiscal_year_end, on='identifier', how='left')

    # select identifier with correct year end
    df = df.loc[df['fiscal_year_end'].isin(['MAR','JUN','SEP','DEC'])]

    # find last fiscal_year_end
    df['fiscal_year_end'] = df['fiscal_year_end'].replace(['MAR','JUN','SEP','DEC'],  # e.g. MAR -> 0331
                                                                    ['0331','0630','0930','1231'])
    df['last_year_end'] = (df['year'] - 1).astype(str) + df['fiscal_year_end']   # 20050331
    df['last_year_end'] = pd.to_datetime(df['last_year_end'], format='%Y%m%d')        # datetime

    # find period_end
    df['period_end'] = df.apply(lambda x: x['last_year_end'] + pd.offsets.MonthEnd(x['frequency_number']*3), axis=1)
    print(df.columns)

    return df.drop(['last_year_end','fiscal_year_end','year','frequency_number'], axis=1)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    y = eps_to_ni_yoy()