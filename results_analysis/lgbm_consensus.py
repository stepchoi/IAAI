from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score
from tqdm import tqdm
from preprocess.ratios import full_period, worldscope
from miscel import date_type, check_dup
from collections import Counter
import os

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

indi_sector = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
               501010, 201020, 502030, 401010, 999999]
indi_industry_new = [11, 20, 30, 35, 40, 45, 51, 60, 65]

idd = '00130H105'
edd = '2003-06-30'
def check_id(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.loc[(df['identifier'] ==idd) & (df['period_end']==edd)].sort_values(['period_end']))
    exit(0)

def download_add_detail(r_name, table_name):
    ''' download from DB TABLE results_lightgbm_stock '''

    print('----------------> update stock results from DB TABLE {}'.format(table_name))

    with engine.connect() as conn:

        if r_name != 'all':     # read DB TABLE results_lightgbm data for given "name"
            result_all = pd.read_sql("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, exclude_fwd, "
                                     "y_type, x_type FROM results_lightgbm WHERE name='{}'".format(r_name), conn)
        else:   # download everything
            result_all = pd.read_sql("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, "
                                     "exclude_fwd, y_type, x_type FROM results_lightgbm", conn)
            result_all = result_all.dropna(subset=['x_type'])
        trial_lgbm = set(result_all['trial_lgbm'])

        try:
            result_stock = pd.read_csv('results_analysis/compare_with_ibes/stocks_all.csv')
            print('local version run - stock_all')
        except:
            # read corresponding part of DB TABLE results_lightgbm_stock
            print('----------> download from DB: stock detail')
            query = text('SELECT * FROM results_lightgbm_stock WHERE (trial_lgbm IN :trial_lgbm)')
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_stock = pd.read_sql(query, conn)
            # print('finish download result_stock', result_stock.info())

    engine.dispose()

    detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')   # map training information to stock data

    return detail_stock

class eps_to_yoy:
    ''' 1. calculate    1. IBES forward NET INCOME =
                        IBES forward EPS * common share outstanding used to calculate EPS (both at T0)
                        2. YoY = (forward NI - actual TTM NI) / current Market Cap
    '''

    def __init__(self):
        '''read IBES original data for calculation'''

        try:    # read ibes data + clean Y
            self.ibes = pd.read_csv('preprocess/ibes_data.csv', usecols = ['identifier','period_end',
                                                                           'EPS1FD12','EPS1TR12'])
            self.ibes.columns = ['identifier', 'period_end', 'eps1fd12', 'eps1tr12']
            self.actual = pd.read_csv('preprocess/clean_ratios.csv', usecols = ['identifier', 'period_end','y_ni'])
            print('local version run - ibes / clean_ratios (actual) ')
        except:
            with engine.connect() as conn:
                self.ibes = pd.read_sql('SELECT identifier, period_end, eps1fd12, eps1tr12 FROM ibes_data', conn)
                self.actual = pd.read_sql('SELECT identifier, period_end, y_ni FROM clean_ratios', conn)
            engine.dispose()

        try:    # read Worldscope Data after cleansing
            self.ws = pd.read_csv('preprocess/quarter_summary_clean.csv', usecols=['identifier','period_end', 'fn_18263', 'fn_8001', 'fn_5192'])
            print('local version run - quarter_summary_clean (with period_end) ')
        except:
            self.ws = worldscope().fill_missing_ws()[['identifier','period_end', 'fn_18263', 'fn_8001', 'fn_5192']]

    def merge_and_calc(self):
        ''' merge worldscope & ibes dataframe and calculate YoY ratios '''

        self.ibes = self.ibes.groupby(['identifier', 'period_end']).mean().reset_index(drop=False)  # for cross listing use average

        self.ibes = date_type(self.ibes)     # convert period_end column to datetime type
        self.actual = date_type(self.actual)
        self.ws = date_type(self.ws)

        # map common share outstanding & market cap to ibes estimations
        self.ibes = self.ibes.merge(self.ws, on=['identifier', 'period_end'])
        self.ibes = self.ibes.merge(self.actual, on=['identifier', 'period_end'])

        # calculate YoY (Y)
        self.ibes = full_period(self.ibes, 'identifier', '%Y-%m-%d')    # order df in chron order

        self.ibes['y_ibes'] = (self.ibes['eps1tr12'].shift(-4) - self.ibes['eps1tr12']) / self.ibes['fn_8001'] * self.ibes['fn_5192']
        self.ibes.loc[self.ibes.groupby('identifier').tail(4).index, 'y_ibes'] = np.nan     # use ibes ttm for Y

        self.ibes['y_consensus'] = (self.ibes['eps1fd12'] - self.ibes['eps1tr12']) * self.ibes['fn_5192'] / self.ibes['fn_8001']     # use ibes fwd & ttm for Y estimation


        self.ibes = label_sector(self.ibes[['identifier', 'period_end', 'y_consensus', 'y_ibes','y_ni']]).dropna(how='any')

        # for name, g in self.ibes.groupby('icb_sector'):
        #     print(name, len(g))

        return self.ibes

def yoy_to_median(yoy):
    ''' 2. convert yoy in qcut format to medians with med_train from training set'''

    try:    # read cut_bins from DB TABLE results_bins_new
        bins_df = pd.read_csv('results_analysis/results_bins_new.csv')
        print('local version run - results_bins ')
    except:
        with engine.connect() as conn:
            bins_df = pd.read_sql('SELECT * from results_bins_new', conn)
            bins_df.to_csv('results_analysis/results_bins_new.csv', index=False)
        engine.dispose()

    def to_median(part_series, cut_bins_dict):
        ''' convert qcut bins to median of each group '''

        cut_bins = cut_bins_dict['cut_bins'].strip('[]').split(',')   # convert string {1, 2, 3....} to list
        med_test = cut_bins_dict['med_train'].strip('[]').split(',')

        cut_bins[0] = -np.inf  # convert cut_bins into [-inf, ... , inf]
        cut_bins[-1] = np.inf
        cut_bins = [float(x) for x in cut_bins]     # convert string in list to float
        med_test = [float(x) for x in med_test]

        arr_q = pd.cut(part_series, bins=cut_bins, labels=False)  # cut original series into 0, 1, .... (bins * n)
        arr_q = arr_q.replace(range(int(cut_bins_dict['qcut_q'])), med_test).values  # replace 0, 1, ... into median

        return arr_q  # return converted Y and median of all groups

    yoy_list = []

    for i in tqdm(range(len(bins_df))):   # roll over all cut_bins used by LightGBM -> convert to median

        part_yoy = yoy.copy()

        if bins_df.iloc[i]['icb_code'] in [0, 1]:   # represent miscellaneous model
            part_yoy['icb_code'] = bins_df.iloc[i]['icb_code']

        elif bins_df.iloc[i]['icb_code'] in indi_sector:
            part_yoy['icb_code'] = part_yoy['icb_sector']

        elif bins_df.iloc[i]['icb_code'] in indi_industry_new:
            part_yoy['icb_code'] = part_yoy['icb_industry']

        part_yoy = part_yoy.loc[(part_yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                                (part_yoy['icb_code'] == bins_df.iloc[i]['icb_code'])].drop(['icb_sector','icb_industry'], axis=1)

        # qcut (and convert to median if applicable) for y_ibes, y_ni, y_ibes_act
        part_yoy['y_consensus_qcut'] = to_median(part_yoy['y_consensus'], cut_bins_dict=bins_df.iloc[i])
        part_yoy['y_ni_qcut'] = to_median(part_yoy['y_ni'], cut_bins_dict=bins_df.iloc[i])
        part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes'], cut_bins_dict=bins_df.iloc[i])
        part_yoy['y_type'] = bins_df.iloc[i]['y_type']

        yoy_list.append(part_yoy)

    return pd.concat(yoy_list, axis=0)

class download:
    ''' download stock / ibes data and convert to qcut_median '''

    def __init__(self, r_name):
        self.r_name = r_name
        self.detail_stock = self.download_stock_data()
        self.yoy_med = self.convert_download_ibes()

    def download_stock_data(self):
        ''' Download 1: download lightgbm results for stocks '''

        try:
            detail_stock = pd.read_csv('results_analysis/compare_with_ibes/stock_{}.csv'.format(self.r_name))
            detail_stock = date_type(detail_stock, date_col='testing_period')
            print('local version run - stock_{}'.format(self.r_name), detail_stock.shape)
        except:
            detail_stock = download_add_detail(self.r_name,'results_lightgbm_stock')
            detail_stock.to_csv('results_analysis/compare_with_ibes/stock_{}.csv'.format(self.r_name), index=False)

        return detail_stock

    def download_ibes(self):
        ''' Download 2: download ibes_data and organize to YoY '''

        try:
            yoy = pd.read_csv('results_analysis/compare_with_ibes/ibes_yoy.csv')
            yoy = date_type(yoy)
            print('local version run - ibes_yoy')
        except:
            yoy = eps_to_yoy().merge_and_calc()
            yoy.to_csv('results_analysis/compare_with_ibes/ibes_yoy.csv', index=False)

        return yoy

    def convert_download_ibes(self):
        ''' convert ibes to median '''

        try:
            yoy_med = pd.read_csv('results_analysis/compare_with_ibes/ibes_median.csv')
            yoy_med = date_type(yoy_med)
            print('local version run - ibes_median')
        except:
            yoy = self.download_ibes()
            yoy_med = yoy_to_median(yoy)                            # STEP2: convert ibes YoY to qcut / median
            yoy_med.to_csv('results_analysis/compare_with_ibes/ibes_median.csv', index=False)

        return yoy_med

    def merge_stock_ibes(self, agg_type='median'):
        ''' combine all prediction together '''

        # convert datetime
        # self.detail_stock['exclude_fwd'] = self.detail_stock['exclude_fwd'].fillna(False)
        self.detail_stock['y_type'] = self.detail_stock['y_type'].fillna('ni')

        print(self.detail_stock.shape)

        # decide base list -> identifier + period_end appeared in both lgbm and rnn models
        lgbm = pd.read_csv('results_analysis/compare_with_ibes/stock_ibes_new industry_only ws -indi space3.csv',
                           usecols=['identifier', 'testing_period'])
        rnn = pd.read_csv('results_analysis/compare_with_ibes/rnn_eps_stock_all.csv',
                          usecols=['identifier', 'testing_period'])
        base_list = pd.merge(lgbm, rnn, on=['identifier', 'testing_period'], how='inner')
        base_list = date_type(base_list, 'testing_period')

        self.detail_stock = self.detail_stock.merge(base_list, on=['identifier', 'testing_period'], how='right')
        print(self.detail_stock.shape)

        if 'entire' in r_name: # for entire
            print('------ convert entire ------')
            self.detail_stock.loc[self.detail_stock['icb_code']==1, 'x_type'] = 'fwdepsqcut-industry_code'
            self.detail_stock.loc[self.detail_stock['icb_code']==2, 'x_type'] = 'fwdepsqcut-sector_code'
            self.detail_stock['icb_code'] = 0

        # print(set(self.detail_stock['x_type']))

        self.detail_stock = self.detail_stock.drop_duplicates(
            subset=['icb_code', 'identifier', 'testing_period', 'cv_number','x_type','y_type'], keep='last')

        # use median/mean for cross listing & multiple cross-validation
        if agg_type == 'mean':
            self.detail_stock = self.detail_stock.groupby(['icb_code', 'identifier', 'testing_period', 'exclude_fwd',
                                                           'x_type','y_type']).mean()['pred'].reset_index(drop=False)
        elif agg_type == 'median':
            self.detail_stock = self.detail_stock.groupby(['icb_code', 'identifier', 'testing_period', 'exclude_fwd',
                                                           'x_type', 'y_type']).median()['pred'].reset_index(drop=False)
        else:
            exit(1)

        self.detail_stock['icb_code'] = self.detail_stock['icb_code'].astype(float)   # convert icb_code to int
        self.yoy_med['icb_code'] = self.yoy_med['icb_code'].astype(float)

        # merge (stock prediction) with (ibes consensus median)
        yoy_merge = self.detail_stock.merge(self.yoy_med, left_on=['identifier', 'testing_period', 'y_type', 'icb_code'],
                                            right_on=['identifier', 'period_end','y_type', 'icb_code'],
                                            suffixes=('_lgbm', '_ibes'))

        # c = yoy_merge.loc[yoy_merge['x_type']=='fwdepsqcut'].values == yoy_merge.loc[yoy_merge['x_type']=='fwdepsqcut-industry_code'].values
        # print(c)
        # print(yoy_merge.columns)
        # print(c.sum(axis=0))
        # exit(0)

        # new = {}
        # new['consensus'] = mean_absolute_error(yoy_merge_1['y_consensus_qcut'], yoy_merge_1['y_ibes_qcut'])
        # new['lgbm'] = mean_absolute_error(yoy_merge_1['pred'], yoy_merge_1['y_ibes_qcut'])
        # print(i, r_name, new)

        # exit(0)

        return label_sector(yoy_merge[['identifier', 'testing_period', 'y_type', 'x_type', 'pred', 'icb_code',
                                       'y_consensus_qcut', 'y_ni_qcut', 'y_ibes_qcut', 'y_ibes', 'y_consensus']]) # 'exclude_fwd -> x_type

class calc_mae_write():

    def __init__(self, yoy_merge, tname=''):
        ''' calculate all MAE and save to local xlsx '''

        self.tname = tname
        # self.merge = yoy_merge
        # self.merge['exclude_fwd'] = self.merge['exclude_fwd'].replace([True, False], ['ex_fwd', 'in_fwd'])
        yoy_merge['icb_code'] = yoy_merge['icb_code'].astype(int).astype(str)
        yoy_merge['icb_type'] = [len(x) for x in yoy_merge['icb_code']]
        yoy_merge['icb_type'] = yoy_merge['icb_type'].astype(str)


        for name, g in yoy_merge.groupby(['y_type', 'icb_type']):

            self.name = name
            self.merge = g

            self.writer = pd.ExcelWriter('results_analysis/compare_with_ibes/mae_{}｜{}.xlsx'.format('_'.join(name),tname))

            self.by_sector().to_excel(self.writer, 'by_sector')
            self.by_industry().to_excel(self.writer, 'by_industry')
            self.by_time().to_excel(self.writer, 'by_time')
            self.average().to_excel(self.writer, 'average')

            print('save to file name: mae_{}｜{}.xlsx'.format('_'.join(name), tname))
            self.writer.save()

    def by_sector(self):
        ''' calculate equivalent per sector MAE '''

        sector_dict = {}
        for name, g in self.merge.groupby(['icb_sector','x_type']): # exclude_fwd -> x_type
            sector_dict[name] = self.part_mae(g)

        df = pd.DataFrame(sector_dict).T.unstack()
        df.columns = ['_'.join(x) for x in df.columns.to_list()]
        return label_sector_name(df)

    def by_industry(self):
        ''' calculate equivalent per industry(new) MAE '''

        industry_dict = {}
        for name, g in self.merge.groupby(['icb_industry', 'x_type']):
            industry_dict[name] = self.part_mae(g)

        df = pd.DataFrame(industry_dict).T.unstack()
        df.columns = ['_'.join(x) for x in df.columns.to_list()]

        return label_industry_name(df)

    def by_time(self):
        ''' calculate equivalent per testing_period MAE '''

        industry_dict = {}

        for name, g in self.merge.groupby(['testing_period', 'x_type']):
            industry_dict[name] = self.part_mae(g)

        df = pd.DataFrame(industry_dict).T.unstack()
        df.columns = ['_'.join(x) for x in df.columns.to_list()]
        print(df)

        return df

    def average(self):
        ''' calculate total MAE '''

        industry_dict = {}

        for name, g in self.merge.groupby(['x_type']):
            industry_dict['_'.join(self.name) + '|' + name] = self.part_mae(g)

        df = pd.DataFrame(industry_dict).T
        print(df.T)

        return df[['lgbm_mae','consensus_mae','lgbm_mse','consensus_mse','lgbm_r2','consensus_r2','consensus_r2_org','len']]

    def part_mae(self, df):
        ''' calculate different mae for groups of sample '''

        dict = {}
        dict['consensus_mae'] = mean_absolute_error(df['y_ibes_qcut'], df['y_consensus_qcut'])
        dict['consensus_mse'] = mean_squared_error(df['y_ibes_qcut'], df['y_consensus_qcut'])
        dict['consensus_r2'] = r2_score(df['y_ibes_qcut'], df['y_consensus_qcut'])
        dict['consensus_r2_org'] = r2_score(df['y_ibes'], df['y_consensus'])
        if 'ibes' in self.name:
            dict['lgbm_mae'] = mean_absolute_error(df['y_ibes_qcut'], df['pred'])
            dict['lgbm_mse'] = mean_squared_error(df['y_ibes_qcut'], df['pred'])
            dict['lgbm_r2'] = r2_score(df['y_ibes_qcut'], df['pred'])
        elif 'ni' in self.name:
            dict['lgbm_mae'] = mean_absolute_error(df['y_ni_qcut'], df['pred'])
            dict['lgbm_mse'] = mean_squared_error(df['y_ni_qcut'], df['pred'])
            dict['lgbm_r2'] = r2_score(df['y_ni_qcut'], df['pred'])
        dict['len'] = len(df)
        return dict

def label_sector(df):
    ''' find sector(6) / industry(2) for each identifier '''

    try:
        icb = pd.read_csv('preprocess/dl_value_universe_icb.csv')
        print('local version run - dl_value_universe_icb')
    except:
        with engine.connect() as conn:
            icb = pd.read_sql("SELECT icb_sector, identifier FROM dl_value_universe WHERE identifier IS NOT NULL", conn)
        engine.dispose()
        icb.to_csv('preprocess/dl_value_universe_icb.csv', index=False)

    icb['icb_industry'] = icb.dropna(how='any')['icb_sector'].astype(str).str[:2].astype(int)
    icb['icb_industry'] = icb['icb_industry'].replace([10, 15, 50, 55], [11, 11, 51, 51])
    icb['icb_sector'] = icb['icb_sector'].mask(~icb['icb_sector'].isin(indi_sector), 999999)

    return df.merge(icb.drop_duplicates(), on=['identifier'], how='left')   # remove dup due to cross listing

def label_sector_name(df):
    ''' label sector name for each icb_sector '''

    with engine.connect() as conn:
        ind_name = pd.read_sql('SELECT * FROM industry_group', conn).set_index(['industry_group_code'])
    engine.dispose()

    ind_name.index = ind_name.index.astype(int)
    df = df.merge(ind_name, left_index=True, right_index=True, how='left')
    # print(df)

    return df

def label_industry_name(df):
    '''label sector name for each icb_industry '''

    with engine.connect() as conn:
        ind_name = pd.read_sql('SELECT * FROM industry_group_2', conn).set_index(['icb_industry'])
    engine.dispose()

    df = df.merge(ind_name, left_index=True, right_index=True, how='left')
    # print(df)

    return df

def combine():
    ''' combine average of different trial to save csv for comparison'''

    os.chdir('results_analysis/compare_with_ibes')

    average = []

    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    for f in files:     # append all average records from files
        if (f[:9]=='mae_ibes_') and (f[-5:]=='.xlsx'):
            f_name = f[11:-5]
            avg = pd.read_excel(f, 'average', index_col='Unnamed: 0')
            avg.index = [x + '|' + f_name for x in avg.index]
            average.append(avg)

    writer = pd.ExcelWriter('#compare_all.xlsx')    # create excel records

    # combine all average records
    avg_df = pd.concat(average, axis=0)
    avg_df = avg_df.filter(sorted(avg_df.columns.to_list())).reset_index()
    avg_df['index'] = avg_df['index'].replace(['1','2','6'],['entire','industry','sector'])
    avg_df = avg_df.set_index(['index'])#.sort_values(['lgbm'])

    def find_col(k):
        return [x for x in avg_df.columns if k in x]

    avg_df[['lgbm_mae','consensus_mae','lgbm_mse','consensus_mse','lgbm_r2',
            'consensus_r2','consensus_r2_org','len']].to_excel(writer, 'average_mae')    # write to files

    print('save to file name: #compare_all.xlsx')
    writer.save()

def compare_by_part():

    ''' (unfinished) compare different trials in each sector / industry '''

    sector = []
    industry = []

    def select(f, name):
        df = pd.read_excel(f, name, index_col='Unnamed: 0')

        new_col = []
        for col in df.columns:
            try:
                s = df[col]
                s = s.rename(f_name + '_' + col[-8:-6])
                new_col.append(s)
            except:
                continue

        return pd.concat(new_col, axis=1)

    # sector.append(select(f, 'by_sector'))
    # industry.append(select(f, 'by_industry'))

    # label_sector_name(pd.concat(sector, axis=1)).to_excel(writer, 'by_sector_lgbm_in')
    # label_industry_name(pd.concat(industry, axis=1)).to_excel(writer, 'by_industry_lgbm_in')

if __name__ == "__main__":

    # df1 = pd.read_csv('results_analysis/compare_with_ibes/stock_ibes_new industry_only ws -indi space.csv')
    # df2 = pd.read_csv('results_analysis/compare_with_ibes/stock_ibes_new industry_only ws -indi space2.csv')
    # print(df1.shape)
    # print(df2.shape)
    #
    # merge_col = ['identifier','qcut_q','icb_code','testing_period','cv_number','exclude_fwd','y_type','x_type']
    # print(df1.loc[df1.duplicated(merge_col, keep=False)])
    # print(df2.loc[df2.duplicated(merge_col, keep=False)])
    #
    # d = pd.merge(df1, df2, on=merge_col, how='left', suffixes=['_1','_2'])
    # print(d.isnull().sum())
    # print(d.loc[d['pred_2'].isnull()])
    #
    # d.loc[d['pred_2'].isnull()].to_csv('#check.csv')
    #
    # exit(0)


    # r_name = 'ibes_sector_only ws'      # name in DB results_lightgbm
    # r_name = 'ibes_new industry_monthly -new'
    r_name_list = ['ibes_sector_all x', 'ibes_new industry_all x -mse', 'ibes_new industry_all x -indi space', 'ibes_sector_only ws -indi space',
                   'ibes_new industry_only ws -indi space3', 'ibes_entire_only ws -smaller space']

    r_name = 'ibes_sector_all x'      # name in DB results_lightgbm

    # for r_name in r_name_list:
    yoy_merge = download(r_name).merge_stock_ibes(agg_type='median')
    calc_mae_write(yoy_merge, tname=r_name)


        # import matplotlib.pyplot as plt
        # yoy_merge['res_consensus'] = yoy_merge['y_consensus_qcut'] - yoy_merge['y_ibes_qcut']
        # yoy_merge['res_pred'] = yoy_merge['pred'] - yoy_merge['y_ibes_qcut']
        # plt.hist(yoy_merge['res_pred'], bins=1000, color='b', alpha=0.5, label='residual_lgbm')
        # plt.hist(yoy_merge['res_consensus'], bins=1000, color='y', alpha=0.5, label='residual_consensus')
        # plt.ylim(0,200)
        # plt.legend()
        # plt.show()
        #
        # exit(0)


    combine()



