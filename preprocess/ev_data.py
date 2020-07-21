from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import xarray

import datetime as dt
from dateutil.relativedelta import relativedelta
from collections import Counter

from miscel import check_dup, date_type
from preprocess.ratios import worldscope, full_period, trim_outlier

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def label_nation_sector(df):
    ''' read files mapping icb_code & market '''

    with engine.connect() as conn:
        markets = pd.read_sql(text("SELECT index_ric, icb_sector, identifier, member_ric FROM dl_value_universe "
                              "WHERE identifier IS NOT NULL"), conn)
    engine.dispose()

    df = pd.merge(df, markets, on=['identifier'])
    df['market'] = df['index_ric'].replace(['0#.CSI300', '0#.N225', '0#.SPX', '0#.HSLI'], ['CH', 'JP', 'US', 'HK'])
    return df.drop(['index_ric'], axis=1)

def find_from_sp500():
    ''' obselete '''

    med = pd.read_csv('preprocess/med_ws.csv')

    small_med = med.loc[med['fn_18100'] < 1e12]

    print(small_med[['member_ric', 'fn_18100']].sort_values('fn_18100').drop_duplicates('member_ric', keep='last'))

def filter_id_med_small():
    ''' filter small market company by size '''

    with engine.connect() as conn:

        # from worldscope_static find company in sector 201030 (phama)
        id_med = pd.read_sql(text("SELECT DISTINCT identifier FROM worldscope_static "
                         "WHERE (field_number = '7040.0') AND (data like '201030%')"), conn)['identifier'].to_list()
        print('Phama Company (201030) available: ', len(id_med))

        # from worldscope_quarter_summary find field 18100 (market cap)
        query = text('SELECT DISTINCT identifier FROM worldscope_quarter WHERE (identifier IN :trial_lgbm) AND '
                     '(year=2020) AND (field_number = 18100) AND (data < 6.1e14)') # UPDATE maximun pt size
        query = query.bindparams(trial_lgbm=tuple(id_med))
        id_med_small = pd.read_sql(query, conn)['identifier'].to_list()
        print('Small Company (EV < 1e12): ', len(id_med_small))

        query_2 = text('SELECT DISTINCT * FROM worldscope_ticker WHERE (identifier IN :trial_lgbm)')
        query_2 = query_2.bindparams(trial_lgbm=tuple(id_med_small))
        name_med_small = pd.read_sql(query_2, conn)
        print('Company with Name: ', len(name_med_small))

    engine.dispose()

    print(name_med_small)

    name_med_small.to_csv('preprocess/small_med_name.csv', index=False)

def download_data_small_med(id_med_small=[]):

    if len(id_med_small) == 0:
        id_med_small = pd.read_csv('preprocess/small_med_name.csv', usecols=['identifier'])['identifier'].to_list()

    with engine.connect() as conn:

        # download all from worldscope_quarter
        query = text('SELECT * FROM worldscope_quarter WHERE (identifier IN :trial_lgbm)')
        query = query.bindparams(trial_lgbm=tuple(id_med_small))

        print('start')

        worldscope_quarter = pd.read_sql(query, conn)
        print(worldscope_quarter)

        query = text('SELECT * FROM worldscope_static WHERE (identifier IN :trial_lgbm)')
        query = query.bindparams(trial_lgbm=tuple(id_med_small))
        worldscope_static = pd.read_sql(query, conn)
        print(worldscope_static)

    engine.dispose()

    if len(id_med_small) == 0:
        worldscope_quarter.to_csv('preprocess/small_med_worldscope_quarter.csv', index=False)
        worldscope_static.to_csv('preprocess/small_med_worldscope_static.csv', index=False)

    return worldscope_quarter, worldscope_static

def org_data_small_med(worldscope_quarter = [], worldscope_static=[]):

    if len(worldscope_quarter) == len(worldscope_static) == 0:
        worldscope_static = pd.read_csv('preprocess/small_med_worldscope_static.csv').drop_duplicates(['identifier', 'field_number'])
        worldscope_quarter = pd.read_csv('preprocess/small_med_worldscope_quarter.csv')

    worldscope_static['field_number'] =  worldscope_static['field_number'].astype(int)
    worldscope_static = worldscope_static.drop_duplicates(['identifier','field_number'], keep='last')

    static_summary = worldscope_static.set_index(['identifier','field_number'])['data'].unstack().reset_index()

    worldscope_quarter = worldscope_quarter.merge(static_summary[['identifier',5352]], on=['identifier'])
    worldscope_quarter = label_period_end(worldscope_quarter)

    quarter_summary = worldscope_quarter.set_index(['identifier', 'period_end',
                                                    'field_number'])['data'].unstack().reset_index()

    if len(worldscope_quarter) == len(worldscope_static) == 0:
        static_summary.to_csv('preprocess/small_med_static_summary.csv', index=False)
        quarter_summary.to_csv('preprocess/small_med_quarter_summary.csv', index=False)

    return static_summary, quarter_summary


def rename_col(static_summary=[], quarter_summary=[]):
    name = pd.read_csv('preprocess/worldscope_field_name.csv')

    if len(static_summary) == len(quarter_summary) == 0:
        quarter_summary = pd.read_csv('preprocess/small_med_quarter_summary.csv')
        static_summary = pd.read_csv('preprocess/small_med_static_summary.csv')
        # print(name['field_number'].astype(str).to_list())

    print(pd.DataFrame(quarter_summary.columns).replace(name['field_number'].astype(str).to_list(), name['field_name'].to_list()))

    quarter_summary.columns = pd.DataFrame(quarter_summary.columns).replace(name['field_number'].astype(str).to_list(), name['field_name'].to_list()).iloc[:,0].to_list()
    static_summary.columns = pd.DataFrame(static_summary.columns).replace(name['field_number'].astype(str).to_list(), name['field_name'].to_list()).iloc[:,0].to_list()

    print(quarter_summary, static_summary)

    if len(static_summary) == len(quarter_summary) == 0:
        quarter_summary.to_csv('preprocess/small_med_quarter_summary_name.csv', index=False)
        static_summary.to_csv('preprocess/small_med_static_summary_name.csv', index=False)

    return static_summary, quarter_summary


def label_period_end(df):
    ''' find fiscal_period_end -> last_year_end for each identifier + frequency_number * 3m '''

    # select identifier with correct year end
    df = df.loc[df[5352].isin(['MAR','JUN','SEP','DEC'])]
    print('1.2. after align year_end: ', df.shape, '# id: ', len(set(df['identifier'])))

    # find last fiscal_year_end
    df[5352] = df[5352].replace(['MAR','JUN','SEP','DEC'], ['0331','0630','0930','1231'])
    df['last_year_end'] = (df['year'] - 1).astype(str) + df[5352]   # 20050331
    df['last_year_end'] = pd.to_datetime(df['last_year_end'], format='%Y%m%d')        # datetime

    # find period_end
    df['period_end'] = df.apply(lambda x: x['last_year_end'] + pd.offsets.MonthEnd(x['frequency_number']*3), axis=1)

    return df.drop(['last_year_end', 5352 ,'year','frequency_number'], axis=1)

def pt_download_organize():
    pt = pd.read_excel('preprocess/small_med/gilead_pt_summary.xlsx', 'public')
    pt_id = pt['identifier'].to_list()

    try:
        static = pd.read_csv('preprocess/small_med/pt_static.csv', index_col='Unnamed: 0')
        quarter = pd.read_csv('preprocess/small_med/pt_quarter.csv', index_col='Unnamed: 0')
        print(static, quarter)

    except:
        quarter, static = download_data_small_med(pt_id)
        static.to_csv('preprocess/small_med/pt_static.csv')
        quarter.to_csv('preprocess/small_med/pt_quarter.csv')

    static_summary, quarter_summary = org_data_small_med(quarter, static)

    static_summary, quarter_summary = rename_col(static_summary, quarter_summary)

    static_summary.to_csv('preprocess/small_med/pt_static_summary_name.csv')
    quarter_summary.to_csv('preprocess/small_med/pt_quarter_summary_name.csv')

def pt_analysis():
    quarter_summary = pd.read_csv('preprocess/small_med/pt_quarter_summary_name.csv')
    static_summary = pd.read_csv('preprocess/small_med/pt_static_summary_name.csv')
    ev = quarter_summary.groupby('identifier').last()[['period_end', '18100']]

    print(ev)



if __name__ == '__main__':

    # pt_analysis(

    # filter_id_med_small()
    download_data_small_med()

