from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt

from miscel import check_dup, date_type

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

class worldscope:

    def __init__(self):

        ''' organize worldscope_quarter_summary data -> identifier + period_end + data'''

        try:
            self.ws = pd.read_csv('preprocess/quarter_summary.csv')     # local version
            print('local version run - quarter_summary')
        except:
            self.ws = pd.read_sql('select * from worldscope_quarter_summary', engine)
            engine.dispose()

        print('1.0. original: ', self.ws.shape, '# id: ', len(set(self.ws['identifier'])))

        self.drop_dup()

    def drop_dup(self):
        ''' drop duplicate records for same identifier & fiscal period, keep the most complete records '''

        self.ws['count'] = pd.isnull(self.ws).sum(1)
        self.ws = self.ws.sort_values(['count']).drop_duplicates(subset=['identifier', 'year', 'frequency_number'],
                                                                 keep='first').drop('count', 1)

        print('1.1. after drop_duplicates: ', self.ws.shape, '# id: ', len(set(self.ws['identifier'])))

    def label_period_end(self):
        ''' find fiscal_period_end -> last_year_end for each identifier + frequency_number * 3m '''

        try:
            fiscal_year_end = pd.read_csv('preprocess/static_fiscal_year_end.csv')
        except:
            fiscal_year_end = pd.read_sql("SELECT identifier, data as fiscal_year_end FROM worldscope_static"
                                          " WHERE field_number = '5352.0'", engine)

        self.ws = pd.merge(self.ws, fiscal_year_end, on='identifier', how='left')

        # select identifier with correct year end
        self.ws = self.ws.loc[self.ws['fiscal_year_end'].isin(['MAR','JUN','SEP','DEC'])]
        print('1.2. after align year_end: ', self.ws.shape, '# id: ', len(set(self.ws['identifier'])))

        # find last fiscal_year_end
        self.ws['fiscal_year_end'] = self.ws['fiscal_year_end'].replace(['MAR','JUN','SEP','DEC'],  # e.g. MAR -> 0331
                                                                        ['0331','0630','0930','1231'])
        self.ws['last_year_end'] = (self.ws['year'] - 1).astype(str) + self.ws['fiscal_year_end']   # 20050331
        self.ws['last_year_end'] = pd.to_datetime(self.ws['last_year_end'], format='%Y%m%d')        # datetime

        # find period_end
        self.ws['period_end'] = self.ws.apply(lambda x: x['last_year_end'] +
                                                        pd.offsets.MonthEnd(x['frequency_number']*3), axis=1)

        return self.ws.drop(['last_year_end','fiscal_year_end','year','frequency_number','fiscal_quarter_end'], axis=1)

    def fill_missing_ws(self):
        ''' fill in missing values by calculating with existing data '''

        ws = self.label_period_end()

        # 1. replace Net Debt (fn_18199) with Total Debt (fn_3255) - Cash & ST investment(fn_2003)
        ws['fn_18199'] = ws['fn_18199'].fillna(ws['fn_3255'] - ws['fn_2001'])

        # 2. replace TTM EBITDA (fn_18309) with EBIT (fn_18308) + DDA (fn_18313)
        ws['fn_18308'] = ws['fn_18308'].fillna(ws['fn_18304'] + ws['fn_18269']) # fill in EBIT = Pretax + interest
        ws['fn_18309'] = ws['fn_18309'].fillna(ws['fn_18308'] + ws['fn_18313'])

        return ws

def calc_divide(ws):
    ''' calculate ratios by dividing two fund items '''

    formula = pd.read_sql('SELECT * FROM ratio_formula', engine)
    engine.dispose()

    # calculate combination fields for roic
    ws['roic_num'] = ws['fn_18309'] - ws['fn_18311']  # TTM EBITDA - TTM Capex
    ws['roic_demon'] = ws['fn_8001'] + ws['fn_18199']  # Market Cap + Net Debt

    num_only_col = ['fn_18267','fn_18158'] # ratios provided by Worldscope

    divide_ratio = ws.filter(['identifier', 'period_end'] + num_only_col) # new table for ratios
    divide_ratio.columns = ['identifier', 'period_end', 'gross_margin','cap_adequacy_ratio']

    for i in range(len(formula)): # fundamental ratio (11) without using ts / fwd
        divide_ratio[formula.iloc[i]['name']] = ws[formula.iloc[i]['field_num']]/ws[formula.iloc[i]['field_demon']]

    return divide_ratio

class calc_ts:

    def __init__(self, ws):
        ''' add full period for ws '''
        self.ts_dict = {'fn_18263': 'ni', 'fn_18262': 'sales', 'fn_18304': 'pretax_margin', 'fn_18265': 'cfps'}
        self.ts_col = list(self.ts_dict.keys())
        self.ws = full_period(ws[['identifier', 'period_end', 'fn_8001'] + self.ts_col], 'identifier')  # add missing records

    def calc_x(self):
        ''' calculate time-series ratios (12) for y-5 ~ y-3 / y-3 ~ y-1 / y-1 ~ y0 '''

        ts_ratio = self.ws[['identifier', 'period_end']]

        def ts_name(name_str): # raname ts ratios e.g. ni_ts01 for y-1 ~ y0
            return [x + '_' + name_str for x in list(self.ts_dict.values())]

        # calculate CAGR for y-5 ~ y-3, y-3 ~ y-1, y-1 ~ y0 -> 4Q = 1Y
        ts_ratio[ts_name('ts01')] = (self.ws[self.ts_col]/self.ws[self.ts_col].shift(4)).sub(1)
        ts_ratio.loc[self.ws.groupby('identifier').head(4).index, ts_name('ts01')] = np.nan # y-1 ~ y0
        ts_ratio[ts_name('ts13')] = np.sqrt(self.ws[self.ts_col]/self.ws[self.ts_col].shift(8)).sub(1).shift(4)
        ts_ratio.loc[self.ws.groupby('identifier').head(12).index, ts_name('ts13')] = np.nan # y-3 ~ y-1
        ts_ratio[ts_name('ts35')] = ts_ratio[ts_name('ts13')].shift(8)
        ts_ratio.loc[self.ws.groupby('identifier').head(20).index, ts_name('ts35')] = np.nan # y-5 ~ y-3

        return ts_ratio

    def calc_y(self):
        ''' calculate y: (NET INCOME of 4Q after - Current NET INCOME) / MARKET CAP '''

        ni = self.ws[['identifier', 'period_end']]
        ni['y_ni'] = (self.ws['fn_18263'].shift(-4) - self.ws['fn_18263']) / self.ws['fn_8001']
        ni['y_rev'] = (self.ws['fn_18262'].shift(-4) - self.ws['fn_18262']) / self.ws['fn_8001']
        ni.loc[ni.groupby('identifier').tail(4).index, ['y_ni', 'y_rev']] = np.nan  # y-1 ~ y0
        return ni.dropna(how='any')

def calc_fwd(ws):
    ''' calculate the forward ratios (2) using DB ibes_data'''

    try:
        ibes = pd.read_csv('preprocess/ibes_data.csv')
        ibes.columns = ['identifier', 'period_end', 'ebd1fd12', 'cap1fd12', 'eps1fd12', 'eps1tr12']
        ibes = date_type(ibes)
        print('local version run - ibes_data')
    except:
        ibes = pd.read_sql('SELECT * FROM ibes_data', engine) # ibes_data is the cleaned with clean_csv.py and uploaded
        engine.dispose()

    ibes = ibes.groupby(['identifier', 'period_end']).mean().reset_index(drop=False)  # for cross listing use average

    ibes['identifier'] = ibes['identifier'].apply(lambda x: x.zfill(9)) # zfill identifiers with leading 0
    ibes_ws = pd.merge(ibes, ws[['identifier','period_end', 'fn_8001','fn_5192','fn_5085', 'roic_demon']],
                       on=['identifier','period_end'])

    def fill_missing_ibse(df):
        ''' fill in missing fwd_roic by replacing missing CAP1FD12 with 0 when company has no history of CAP1FD12'''

        non_cap_comp = set(df['identifier']) - set(df.dropna(subset = ['cap1fd12'])['identifier'])
        df.loc[df['identifier'].isin(non_cap_comp), 'cap1fd12'] = \
            df.loc[df['identifier'].isin(non_cap_comp), 'cap1fd12'].fillna(0)
        return df

    ibes_ws = fill_missing_ibse(ibes_ws)                                        # calculate IBES TTM as Y
    ibes_ws = full_period(ibes_ws, 'identifier')
    ibes_ws['y_ibes'] = (ibes_ws['eps1tr12'].shift(-4) - ibes_ws['eps1tr12']) / ibes_ws['fn_8001'] * ibes_ws['fn_5192']
    ibes_ws.loc[ibes_ws.groupby('identifier').tail(4).index, 'y_ibes'] = np.nan


    ibes_ws['fwd_ey'] = ibes_ws['eps1fd12'] / ibes_ws['fn_5085']
    ibes_ws['fwd_roic'] = (ibes_ws['ebd1fd12'] - ibes_ws['cap1fd12']) / ibes_ws['roic_demon']

    return ibes_ws[['identifier','period_end','fwd_ey','fwd_roic','y_ibes']]

def full_period(df, index_col, date_format=None):
    ''' add NaN for missing records to facilitate time_series ratios calculation (worldscope & stock_return)'''

    start = dt.datetime(1998, 3, 31)
    date_list = [start + pd.offsets.MonthEnd(x * 3) for x in range(90)]  # create list of date for sampling period

    if date_format != None:
        df['period_end'] = pd.to_datetime(df['period_end'], format=date_format)

    full_period = pd.DataFrame(columns=set(df[index_col]), index=date_list)
    full_period = full_period.unstack().reset_index(drop=False).iloc[:, :2]
    full_period.columns = [index_col, 'period_end']
    df_full_period = pd.merge(full_period, df, on=[index_col, 'period_end'], how='left')

    print('# full records length: {}={}*{} '.format(len(df_full_period), len(set(df[index_col])), len(date_list)))

    return df_full_period

def trim_outlier(df, prc=0.01):
    ''' assign a max value for the 99% percentile to replace inf'''

    df_nan = df.replace([np.inf, -np.inf], np.nan)
    try:
        pmax = df_nan.quantile(q=(1 - prc), axis=0)
        df = df.mask(df > pmax, pmax, axis=1)
    except:
        pmax = df_nan.quantile(q=(1 - prc))
        df = df.mask(df > pmax, pmax)
    return df

if __name__ == '__main__':

    # 1. organize worldscope_quarter_summary
    # try:
    #     ws = pd.read_csv('ws.csv')
    #     print('local version run - ws')
    # except:
    ws = worldscope().fill_missing_ws()
    # ws.to_csv('preprocess/quarter_summary_clean.csv', index=False)
    # exit(0)

    # def check_ws_ratios(ws):
    #     ''' check 3 ratios calculated by worldscope - gross margin, pretax margin, D/A '''
    #
    #     ws['new_gross_margin'] = (ws['fn_18262'] - ws['fn_18312'] - ws['fn_18313']) / ws['fn_18262']
    #     ws['new_pretax_margin'] = ws['fn_18271'] / ws['fn_18262']
    #     ws['new_d_a'] = ws['fn_3255'] / ws['fn_2999']
    #
    #     org = ['fn_18267', 'fn_18304', 'fn_8236']
    #     new = ['new_gross_margin', 'new_pretax_margin', 'new_d_a']
    #
    #     for i in range(3):
    #         diff = (ws[org[i]] - ws[new[i]]*10e7).dropna()
    #         print(org[i], new[i])
    #         print(diff.describe())
    #
    # check_ws_ratios(ws)

    # 2. calculate ratios
    divide_ratio = calc_divide(ws)
    # print(divide_ratio.isnull().sum())
    # ws.merge(divide_ratio, on=['identifier','period_end']).to_csv('#check_ratios.csv')
    print('2.1. divide_ratio df shape', divide_ratio.shape)

    ts_ratio_class = calc_ts(ws)
    ts_ratio = ts_ratio_class.calc_x()
    y = ts_ratio_class.calc_y().dropna(how='any')
    print('2.2. worldscope_ts_ratio df shape', ts_ratio.shape)
    print('2.3. y_ni df shape', y.shape)

    ibes_ratio = calc_fwd(ws)
    print('2.4. ibes_ratio df shape', ibes_ratio.shape)

    stock_ratio = pd.read_csv('preprocess/stock_ratios.csv')
    stock_ratio['period_end'] = pd.to_datetime(stock_ratio['period_end'], format='%Y-%m-%d')
    print('2.5. stock_ratio df shape', stock_ratio.shape)

    fund_ratio = pd.merge(divide_ratio, ts_ratio, on=['identifier', 'period_end'], how='left')
    fund_ratio = pd.merge(fund_ratio, ibes_ratio, on=['identifier', 'period_end'], how='left')
    fund_ratio = pd.merge(fund_ratio, stock_ratio, on=['identifier', 'period_end'], how='left')
    fund_ratio = pd.merge(fund_ratio, y, on=['identifier', 'period_end'], how='right')
    print('3.1. all ratios combined df shape', fund_ratio.shape)

    # convert inf (i.e. divided by 0) to 99%
    fund_ratio.iloc[:,2:-1] = trim_outlier(fund_ratio.iloc[:,2:-1])
    # print(fund_ratio.describe().T[['max']])

    fund_ratio.drop_duplicates().to_csv('preprocess/clean_ratios.csv', index=False)
    print(fund_ratio.columns, fund_ratio.shape)
    exit(0)

    with engine.connect() as conn:
        fund_ratio.drop_duplicates().to_sql('clean_ratios', engine, index=False, if_exists='replace')
    engine.dispose()

