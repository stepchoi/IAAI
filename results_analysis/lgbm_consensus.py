from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score
from tqdm import tqdm
from preprocess.x_ratios import full_period, worldscope
from miscel import date_type, check_dup, reorder_col
from collections import Counter
import os

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

indi_sector = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
               501010, 201020, 502030, 401010, 999999]
indi_industry_new = [11, 20, 30, 35, 40, 45, 51, 60, 65]

col_name = ['trial_lgbm', 'name', 'qcut_q', 'icb_code', 'testing_period', 'cv_number', 'exclude_fwd', 'y_type', 'x_type']

compare_using_old_ibes = False

idd = '00130H105'
edd = '2003-06-30'
def check_id(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.loc[(df['identifier'] ==idd) & (df['period_end']==edd)].sort_values(['period_end']))
    exit(0)

def download_add_detail(r_name, table_name):
    ''' download from DB TABLE results_lightgbm_stock '''

    print('----------------> update stock results from DB TABLE {}'.format(table_name))
    if tname == 'xgboost':
        col_name.append('grow_policy')

    with engine.connect() as conn:

        if r_name != 'all':     # read DB TABLE results_lightgbm data for given "name"
            query = text("SELECT {} FROM results_{} WHERE name='{}'".format(', '.join(col_name), tname, r_name))
        else:   # download everything
            query = text("SELECT {} FROM results_lightgbm").format(', '.join(col_name))
        result_all = pd.read_sql(query, conn)
        trial_lgbm = set(result_all['trial_lgbm'])

        print(result_all)

        try:
            result_stock = pd.read_csv('results_analysis/compare_with_ibes/stocks_all.csv')
            print('local version run - stock_all')
        except:
            # read corresponding part of DB TABLE results_lightgbm_stock
            print('----------> download from DB: stock detail')
            query = text('SELECT * FROM results_{}_stock WHERE (trial_lgbm IN :trial_lgbm)'.format(tname))
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_stock = pd.read_sql(query, conn)
            # print('finish download result_stock', result_stock.info())

    engine.dispose()

    detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')   # map training information to stock data
    print(detail_stock)

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
            if compare_using_old_ibes == True:
                self.ibes = pd.read_csv('ibes_data_old.csv')

            self.ibes.columns = ['identifier', 'period_end', 'eps1fd12', 'eps1tr12']
            self.ibes_qoq = pd.read_csv('preprocess/ibes_data_qoq.csv')
            self.actual = pd.read_csv('preprocess/clean_ratios.csv', usecols = ['identifier', 'period_end','y_ni'])
            print('local version run - ibes / clean_ratios (actual) ')
        except:
            with engine.connect() as conn:
                self.ibes = pd.read_sql('SELECT identifier, period_end, eps1fd12, eps1tr12 FROM ibes_data', conn)
                self.ibes_qoq = pd.read_sql('SELECT * FROM ibes_data_qoq', conn)
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
        self.ibes_qoq = self.ibes_qoq.groupby(['identifier', 'period_end']).mean().reset_index(drop=False)  # for cross listing use average

        # map common share outstanding & market cap to ibes estimations
        self.ibes = date_type(self.ibes).merge(date_type(self.ws), on=['identifier', 'period_end'])
        self.ibes = self.ibes.merge(date_type(self.actual), on=['identifier', 'period_end'])

        # calculate YoY (Y)
        if compare_using_old_ibes == True:
            self.ibes = full_period(self.ibes, 'identifier', '%Y-%m-%d')    # order df in chron order
            self.ibes['eps1fd12_1'] = self.ibes['eps1fd12'].shift(-4)
            self.ibes.loc[self.ibes.groupby('identifier').tail(4).index, 'eps1fd12_1'] = np.nan

        self.ibes['y_ibes'] = (self.ibes['eps1tr12'].shift(-4) - self.ibes['eps1tr12']) / self.ibes['fn_8001'] * self.ibes['fn_5192']
        self.ibes.loc[self.ibes.groupby('identifier').tail(4).index, 'y_ibes'] = np.nan     # use ibes ttm for Y

        self.ibes['y_consensus'] = (self.ibes['eps1fd12'] - self.ibes['eps1tr12']) * self.ibes['fn_5192'] / self.ibes['fn_8001']     # use ibes fwd & ttm for Y estimation
        self.ibes = label_sector(self.ibes).dropna(how='any')
        self.ibes = self.ibes.merge(date_type(self.ibes_qoq), on=['identifier', 'period_end'], how='outer')  # add qoq to consensus

        return self.ibes

def yoy_to_median(yoy):
    ''' 2. convert yoy in qcut format to medians with med_train from training set'''

    try:    # read cut_bins from DB TABLE results_bins_new
        bins_df = pd.read_csv('results_analy1sis/results_bins_new.csv')
        print('local version run - results_bins ')
    except:
        with engine.connect() as conn:
            bins_df = pd.read_sql('SELECT * from results_bins_new', conn)
            bins_df.to_csv('results_analysis/results_bins_new.csv', index=False)
        engine.dispose()

    print(set(bins_df['label']))

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

        if bins_df.iloc[i]['icb_code'] in [0]:   # represent miscellaneous model
            part_yoy['icb_code'] = bins_df.iloc[i]['icb_code']

        elif bins_df.iloc[i]['icb_code'] in indi_sector:
            part_yoy['icb_code'] = part_yoy['icb_sector']

        elif bins_df.iloc[i]['icb_code'] in indi_industry_new:
            part_yoy['icb_code'] = part_yoy['icb_industry']
        else:
            continue

        part_yoy = part_yoy.loc[(part_yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                                (part_yoy['icb_code'] == bins_df.iloc[i]['icb_code'])].drop(['icb_sector','icb_industry'], axis=1)

        if bins_df.iloc[i]['y_type'] == 'ibes':     # qcut and convert to median for ibes / ibes_qoq
            part_yoy['y_consensus_qcut'] = to_median(part_yoy['y_consensus'], cut_bins_dict=bins_df.iloc[i])
            part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes'], cut_bins_dict=bins_df.iloc[i])
        elif bins_df.iloc[i]['y_type'] == 'ibes_qoq':
            part_yoy['y_consensus_qcut'] = to_median(part_yoy['y_consensus_qoq'], cut_bins_dict=bins_df.iloc[i])
            part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes_qoq'], cut_bins_dict=bins_df.iloc[i])
        else:
            continue

        part_yoy['y_type'] = bins_df.iloc[i]['y_type']
        part_yoy['label'] = bins_df.iloc[i]['label']

        yoy_list.append(part_yoy)

    yoy_median = pd.concat(yoy_list, axis=0).dropna(subset=['y_consensus_qcut','y_ibes_qcut'], how='all')

    return yoy_median

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

        detail_stock['identifier'] = [x.zfill(9) for x in detail_stock['identifier']]   # update identifier for results
        return detail_stock

    def download_ibes(self):
        ''' Download 2: download ibes_data and organize to YoY '''

        try:
            if compare_using_old_ibes == True:
                exit(0)
            yoy = pd.read_csv('results_analysis/compare_with_ibes/ibes_yoy.csv')
            yoy = date_type(yoy)
            print('local version run - ibes_yoy')
        except:
            yoy = eps_to_yoy().merge_and_calc()
            if compare_using_old_ibes != True:
                yoy.to_csv('results_analysis/compare_with_ibes/ibes_yoy.csv', index=False)

        return yoy

    def convert_download_ibes(self):
        ''' convert ibes to median '''

        try:
            if compare_using_old_ibes == True:
                exit(0)
            yoy_med = pd.read_csv('results_analysis/compare_with_ibes/ibes_median.csv')
            yoy_med = date_type(yoy_med)
            print('local version run - ibes_median')
        except:
            yoy = self.download_ibes()
            yoy_med = yoy_to_median(yoy)                            # STEP2: convert ibes YoY to qcut / median

            if compare_using_old_ibes != True:
                yoy_med.to_csv('results_analysis/compare_with_ibes/ibes_median.csv', index=False)

        return yoy_med

    def merge_stock_ibes(self, agg_type='median'):
        ''' combine all prediction together '''

        self.detail_stock = combine_5_cv_results(self.detail_stock, fill_non_columns, agg_type=agg_type)     # combine 5 results
        self.yoy_med['icb_code'] = self.yoy_med['icb_code'].astype(float)

        # merge (stock prediction) with (ibes consensus median)
        yoy_merge = self.detail_stock.merge(self.yoy_med, left_on=['identifier', 'testing_period', 'y_type', 'icb_code','label'],
                                            right_on=['identifier', 'period_end','y_type', 'icb_code','label'],
                                            suffixes=('_lgbm', '_ibes'))

        return label_sector(yoy_merge)

def fill_non_columns(df):
    ''' fill in columns not exists for certain results table / name for combination '''

    # transitory process: add "market" based on "name" -> all used normal is ok (we no longer keep track of consensus results)
    df['market'] = 'normal'

    df['y_type'] = df['y_type'].fillna('ni')    # y_type default as ni
    df['label'] = 'lgbm_' + df['market']        # market label to merge cut_bins
    df = df.loc[df['icb_code'] == 0]            # use only icb_code = 0, no industry/sector code

    return df

def combine_5_cv_results(df, fill_non_columns=None, agg_type='median'):
    ''' combine results of all 5 cross validation to mean/median '''

    if fill_non_columns != None:    # fill non type for df
        fill_non_columns(df)

    agg_col = ['icb_code', 'identifier', 'testing_period', 'exclude_fwd', 'x_type', 'y_type', 'label']
    
    df = df.drop_duplicates(subset=['cv_number'] + agg_col, keep='last')  # use last results for same type training (i.e. first one might be test run)

    if agg_type == 'mean':  # use median/mean for cross listing & multiple cross-validation
        df = df.groupby(agg_col).mean()['pred'].reset_index(drop=False)
    elif agg_type == 'median':
        df = df.groupby(agg_col).median()['pred'].reset_index(drop=False)
    else:
        exit(1)

    df['icb_code'] = df['icb_code'].astype(float)  # convert icb_code to int
    return df

def combine_market_industry_results():
    ''' combine market models + industry models for final results in new config IV '''

    industry = pd.read_csv('results_analysis/compare_with_ibes/stock_ibes_new industry_only ws -indi space3.csv')
    us = pd.read_csv('results_analysis/compare_with_ibes/stock_3b_country_partition_mae.csv')

    industry = combine_5_cv_results(industry, fill_non_columns)     # combine 5 cv results
    us = combine_5_cv_results(us, fill_non_columns)

    industry['icb_code'] = 0
    us['icb_code'] = 0
    both = pd.merge(us, industry, on=['icb_code', 'identifier', 'testing_period', 'exclude_fwd', 'x_type', 'y_type',
                                      'label'],suffixes=('_mkt', '_ind'), how='inner')
    both['pred'] = both[['pred_mkt', 'pred_ind']].mean(axis=1)

    yoy_med = pd.read_csv('results_analysis/compare_with_ibes/ibes_median.csv')
    yoy_merge = both.merge(yoy_med,left_on=['identifier', 'testing_period', 'y_type', 'icb_code', 'label'],
                                    right_on=['identifier', 'period_end', 'y_type', 'icb_code', 'label'],
                                    suffixes=('_lgbm', '_ibes'))

    print(both)
    yoy_merge = label_sector(yoy_merge)

    calc_mae_write(date_type(yoy_merge, 'testing_period'), r_name='', csv_name='config_4')

class calc_mae_write():

    def __init__(self, yoy_merge, r_name, tname='', base_list_type='all', csv_name=''):
        ''' calculate all MAE and save to local xlsx '''

        # decide base list (for non qoq) -> identifier + period_end appeared in both lgbm and rnn models
        if not 'qoq' in r_name:

            if base_list_type == 'all':
                lgbm = pd.read_csv('results_analysis/compare_with_ibes/dense_stock_mini_tune15_re -code 0 -exclude_fwd True.csv',
                                   usecols=['identifier', 'testing_period'])
            elif base_list_type == 'sp':
                lgbm = pd.read_csv('results_analysis/compare_with_ibes/stock_ibes_industry -sp500.csv',
                                   usecols=['identifier', 'testing_period'])

            rnn = pd.read_csv('results_analysis/compare_with_ibes/rnn_eps_stock_all.csv',
                              usecols=['identifier', 'testing_period'])
            base_list = pd.merge(lgbm, rnn, on=['identifier', 'testing_period'], how='inner').drop_duplicates()
            base_list = date_type(base_list, 'testing_period')
            yoy_merge = yoy_merge.merge(base_list, on=['identifier', 'testing_period'], how='inner')

        # if not 'qoq' in r_name:     # add real eps prediction
        #     yoy_merge['pred_eps'] = yoy_merge['pred'] * yoy_merge['fn_8001'] / yoy_merge['fn_5192'] + yoy_merge['eps1tr12']
        # else:
        #     yoy_merge['pred_eps'] = yoy_merge['pred'] * yoy_merge['fn_8001'] / yoy_merge['fn_5192'] + yoy_merge['i0eps']
        #
        # yoy_merge = full_period(yoy_merge, 'identifier')    # find actual eps for next q/y
        # yoy_merge['eps_nexty'] = yoy_merge['eps1tr12'].shift(-4)
        # yoy_merge.loc[yoy_merge.groupby('identifier').tail(4).index, 'eps_nexty'] = np.nan # last quarter qoq
        # yoy_merge['eps_nextq'] = yoy_merge['i0eps'].shift(-1)
        # yoy_merge.loc[yoy_merge.groupby('identifier').tail(1).index, 'eps_nextq'] = np.nan # last quarter qoq
        # yoy_merge = yoy_merge.dropna(how='any')

        yoy_merge['icb_code'] = yoy_merge['icb_code'].astype(int).astype(str)
        yoy_merge['icb_type'] = [len(x) for x in yoy_merge['icb_code']]
        yoy_merge['icb_type'] = yoy_merge['icb_type'].astype(str)

        for name, g in yoy_merge.groupby(['y_type', 'icb_type']):

            self.name = name
            self.merge = g

            if csv_name != '':
                self.writer = pd.ExcelWriter('results_analysis/compare_with_ibes/mae_new_{}.xlsx'.format(csv_name))
            elif compare_using_old_ibes == True:
                self.writer = pd.ExcelWriter('results_analysis/compare_with_ibes/mae_old_ibes.xlsx')
            else:
                self.writer = pd.ExcelWriter('results_analysis/compare_with_ibes/mae_{}｜{}.xlsx'.format('_'.join(name),tname))

            self.by_sector().to_excel(self.writer, 'by_sector')
            self.by_industry().to_excel(self.writer, 'by_industry')
            self.by_time().to_excel(self.writer, 'by_time')
            self.by_nation().to_excel(self.writer, 'by_nation')
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

    def by_nation(self):
        ''' calculate equivalent per testing_period MAE '''

        industry_dict = {}

        def label_nation_sector(df):
            ''' read files mapping icb_code & market '''

            with engine.connect() as conn:
                markets = pd.read_sql("SELECT index_ric, icb_sector, identifier FROM dl_value_universe "
                                      "WHERE identifier IS NOT NULL", conn)
            engine.dispose()

            ratios = pd.merge(df, markets, on=['identifier'])
            ratios['market'] = ratios['index_ric'].replace(['0#.CSI300', '0#.N225', '0#.SPX', '0#.HSLI'],
                                                           ['CH', 'JP', 'US', 'HK'])
            return ratios.drop(['index_ric'], axis=1)

        merge_nation = label_nation_sector(self.merge)

        for name, g in merge_nation.groupby(['market', 'x_type']):
            industry_dict[name] = self.part_mae(g)

        df = pd.DataFrame(industry_dict).T.unstack()
        df.columns = ['_'.join(x) for x in df.columns.to_list()]
        print(df)

        return df

    def average(self):
        ''' calculate total MAE '''

        industry_dict = {}

        for name, g in self.merge.groupby(['x_type']):
            print(name, len(g))
            industry_dict['_'.join(self.name) + '|' + name] = self.part_mae(g)

        df = pd.DataFrame(industry_dict).T
        print(df.T)

        return df

    def part_mae(self, df):
        ''' calculate different mae for groups of sample '''

        def mean_absolute_percentage_error(y_true, y_pred):
            diff = np.abs((y_true - y_pred) / y_true)
            diff = diff[diff < 1E308]
            return np.mean(diff) * 100

        def median_absolute_error(y_true, y_pred):
            return np.median(np.abs(y_true - y_pred))

        # print(r2_score(df['y_ibes'], df['y_ibes_qcut']))

        dict = {}
        # dict['consensus_mae'] = mean_absolute_error(df['y_ibes_qcut'], df['y_consensus_qcut'])      # after qcut metrices - consensus
        # dict['consensus_mse'] = mean_squared_error(df['y_ibes_qcut'], df['y_consensus_qcut'])
        dict['consensus_r2'] = r2_score(df['y_ibes_qcut'], df['y_consensus_qcut'])

        # dict['lgbm_mae'] = mean_absolute_error(df['y_ibes_qcut'], df['pred'])   # after qcut metrices - lgbm
        # dict['lgbm_mse'] = mean_squared_error(df['y_ibes_qcut'], df['pred'])
        dict['lgbm_r2'] = r2_score(df['y_ibes_qcut'], df['pred'])

        if 'ibes' in self.name:     # calculate when y == ibes (yoy)
            dict['consensus_mae_org'] = mean_absolute_error(df['y_ibes'], df['y_consensus'])    # before qcut metrices - consensus
            dict['consensus_medae_org'] = median_absolute_error(df['y_ibes'], df['y_consensus'])
            dict['consensus_mse_org'] = mean_squared_error(df['y_ibes'], df['y_consensus'])
            dict['consensus_r2_org'] = r2_score(df['y_ibes'], df['y_consensus'])
            # dict['consensus_mae_eps'] = mean_absolute_error(df['eps_nexty'], df['eps1fd12'])
            # dict['consensus_mape_eps'] = mean_absolute_percentage_error(df['eps_nexty'], df['eps1fd12'])

            dict['lgbm_mae_org'] = mean_absolute_error(df['y_ibes'], df['pred'])    # before qcut metrices - lgbm
            dict['lgbm_medae_org'] = median_absolute_error(df['y_ibes'], df['pred'])
            dict['lgbm_mse_org'] = mean_squared_error(df['y_ibes'], df['pred'])
            dict['lgbm_r2_org'] = r2_score(df['y_ibes'], df['pred'])
            # dict['lgbm_mae_eps'] = mean_absolute_error(df['eps_nexty'], df['pred_eps'])
            # dict['lgbm_mape_eps'] = mean_absolute_percentage_error(df['eps_nexty'], df['pred_eps'])

        elif 'ibes_qoq' in self.name:   # calculate when y == ibes qoq
            dict['consensus_mae_org'] = mean_absolute_error(df['y_ibes_qoq'], df['y_consensus_qoq'])
            dict['consensus_medae_org'] = median_absolute_error(df['y_ibes_qoq'], df['y_consensus_qoq'])
            dict['consensus_mse_org'] = mean_squared_error(df['y_ibes_qoq'], df['y_consensus_qoq'])
            dict['consensus_r2_org'] = r2_score(df['y_ibes_qoq'], df['y_consensus_qoq'])
            # dict['consensus_mae_eps'] = mean_absolute_error(df['eps_nextq'], df['epsi1md'])
            # dict['consensus_mape_eps'] = mean_absolute_percentage_error(df['eps_nextq'], df['epsi1md'])

            dict['lgbm_mae_org'] = mean_absolute_error(df['y_ibes_qoq'], df['pred'])    # before qcut metrices - lgbm
            dict['lgbm_medae_org'] = median_absolute_error(df['y_ibes_qoq'], df['pred'])
            dict['lgbm_mse_org'] = mean_squared_error(df['y_ibes_qoq'], df['pred'])
            dict['lgbm_r2_org'] = r2_score(df['y_ibes_qoq'], df['pred'])
            # dict['lgbm_mae_eps'] = mean_absolute_error(df['eps_nextq'], df['pred_eps'])
            # dict['lgbm_mape_eps'] = mean_absolute_percentage_error(df['eps_nextq'], df['pred_eps'])

        # elif 'ni' in self.name:
        #     dict['lgbm_mae'] = mean_absolute_error(df['y_ni_qcut'], df['pred'])
        #     dict['lgbm_mse'] = mean_squared_error(df['y_ni_qcut'], df['pred'])
        #     dict['lgbm_r2'] = r2_score(df['y_ni_qcut'], df['pred'])
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

    avg_df[['lgbm_mae_org','consensus_mae_org','lgbm_mse_org','consensus_mse_org','lgbm_r2_org',
            'consensus_r2_org','consensus_r2_org','len']].to_excel(writer, 'average_mae')    # write to files

    print('save to file name: #compare_all.xlsx')
    writer.save()

if __name__ == "__main__":

    # combine_market_industry_results()
    # exit(0)

    r_name = 'ibes_new industry_only ws -indi space3'
    # r_name = 'ibes_new industry_all x -indi space'
    # r_name = 'ibes_entire_only ws -smaller space'
    # r_name = 'ibes_industry -sp500'
    # r_name = 'sp500_entire'

    # r_name = 'xgb ind2 -sample_type industry -x_type fwdepsqcut'
    # r_name = 'xgb ind4 -sample_type industry -x_type ni'
    # r_name = 'xgb ind_all_tuning -sample_type industry -x_type ni'
    # r_name = 'xgb tune_indi -sample_type industry -x_type ni'
    # r_name = 'xgb tryrun -sample_type entire -x_type fwdepsqcut'

    # r_name = 'ibes_qoq_tune10_ind2'
    # r_name = 'ibes_qoq_tune10_ind3'

    # r_name = 'ibes_new industry_all x -mse'
    r_name = 'rounding_ind_ex'
    r_name = 'optimize_r2_industry'

    r_name = 'mse_tune_entire'      # for mse tuning

    r_name = '3b_country_partition_mae'     # for IIIb country partitions
    r_name = 'tune_mse_ind3'
    tname = 'lightgbm'


    # r_name = 'mse_ex_ind_rounding_tune2'        # worse than qcut -> stop
    # r_name = 'mse_ex_ind_tune6'
    # tname = 'xgboost'

    yoy_merge = download(r_name).merge_stock_ibes(agg_type='median')
    calc_mae_write(yoy_merge, r_name, tname=r_name)

    if compare_using_old_ibes != True:
        combine()



