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
def yoy_to_median(yoy):

    with engine.connect() as conn:
        bins_df = pd.read_sql('SELECT * FROM results_bins', conn)
        # result_lgbm = pd.read_sql('SELECT * FROM results_lightgbm_stock', conn)
    engine.dispose()

    # print(result_lgbm)


    for i in range(len(bins_df)):
        print(bins_df.iloc[i])

        part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                           (yoy['icb_sector'] == bins_df.iloc[i]['icb_code'])]

        part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes'], cut_bins=bins_df.iloc[i]['cut_bins'],
                                            qcut_q=bins_df.iloc[i]['qcut_q'], med_test=bins_df.iloc[i]['med_test'])

        part_yoy.to_csv('#check_ibes_qcut.csv', index=False)
        exit(0)

def to_median(arr, cut_bins, qcut_q, med_test):
    ''' convert qcut bins to median of each group '''

    cut_bins = cut_bins.strip('{}').split(',')
    med_test = med_test.strip('{}').split(',')

    cut_bins[0] = -np.inf  # convert cut_bins into [-inf, ... , inf]
    cut_bins[-1] = np.inf
    cut_bins = [float(x) for x in cut_bins]
    med_test = [float(x) for x in med_test]
    print(cut_bins, med_test)

    print(arr)
    arr_q = pd.cut(arr, bins=cut_bins, labels=False)         # cut original series into 0, 1, .... (bins * n)
    print(arr_q)
    arr_new = arr_q.replace(range(int(qcut_q)), med_test).values  # replace 0, 1, ... into median
    print(arr_new)

    return arr_new                                           # return converted Y and median of all groups

def eps_to_yoy():
    ''' calculate 1. IBES forward NET INCOME =
                     IBES forward EPS * common share outstanding used to calculate EPS (both at T0)
                  2. YoY = (forward NI - actual TTM NI) / current Market Cap
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
        # ws.to_csv('preprocess/ws_ibes.csv', index=False)

    ibes['period_end'] = pd.to_datetime(ibes['period_end'], format='%Y-%m-%d')
    ibes.columns = ['identifier', 'period_end', 'eps1fd12']

    # map common share outstanding & market cap to ibes estimations
    ws = label_period_end(ws)
    ibes = ibes.merge(ws, on=['identifier', 'period_end'])

    # calculate YoY (Y)
    ibes['fwd_ni'] = ibes['eps1fd12'] * ibes['share_osd']
    ibes['y_ibes'] = (ibes['fwd_ni'] - ibes['fn_18263']) / ibes['fn_8001']

    # ibes['act_ni'] = ibes['fn_18263'].shift(-4)
    # ibes.loc[ibes.groupby('identifier').tail(4).index, 'act_ni'] = np.nan
    # ibes.to_csv('#check_ibes_ni.csv', index=False)

    return label_sector(ibes[['identifier', 'period_end', 'y_ibes']]).dropna(how='any')

def label_period_end(df):
    ''' find fiscal_period_end -> last_year_end for each identifier + frequency_number * 3m '''

    try:
        fiscal_year_end = pd.read_csv('preprocess/static_fiscal_year_end.csv')
        print('local version run - static_fiscal_year_end')
    except:
        fiscal_year_end = pd.read_sql("SELECT identifier, data as fiscal_year_end FROM worldscope_static"
                                      " WHERE field_number = '5352.0'", engine)

    df = pd.merge(df, fiscal_year_end, on='identifier', how='left')
    df = df.loc[df['fiscal_year_end'].isin(['MAR','JUN','SEP','DEC'])]  # select identifier with correct year end
    df['fiscal_year_end'] = df['fiscal_year_end'].replace(['MAR','JUN','SEP','DEC'], ['0331','0630','0930','1231'])

    df['last_year_end'] = (df['year'] - 1).astype(str) + df['fiscal_year_end']   # find last fiscal_year_end e.g. 20050331
    df['last_year_end'] = pd.to_datetime(df['last_year_end'], format='%Y%m%d')   # convert 20050331 to datetime

    # find period_end by adding quarter pasted since last_year_end
    df['period_end'] = df.apply(lambda x: x['last_year_end'] + pd.offsets.MonthEnd(x['frequency_number']*3), axis=1)

    return df.drop(['last_year_end','fiscal_year_end','year','frequency_number'], axis=1)

def label_sector(df):
    ''' find sector for each identifier '''

    with engine.connect() as conn:
        icb = pd.read_sql("SELECT icb_sector, identifier FROM dl_value_universe WHERE identifier IS NOT NULL", conn)
    engine.dispose()

    return df.merge(icb, on=['identifier'])

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)


    yoy = eps_to_yoy()
    print(yoy)

    yoy_to_median(yoy)
