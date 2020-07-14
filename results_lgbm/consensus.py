from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
from preprocess.ratios import full_period, worldscope
from miscel import date_type, check_dup
from collections import Counter
import os

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def download_add_detail(r_name, table_name):
    ''' download from DB TABLE results_lightgbm_stock '''

    print('----------------> update stock results from DB TABLE {}'.format(table_name))

    with engine.connect() as conn:

        # read DB TABLE results_lightgbm data for given "name"
        result_all = pd.read_sql("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, exclude_fwd "
                     "FROM results_lightgbm WHERE name='{}'".format(r_name), conn)
        trial_lgbm = set(result_all['trial_lgbm'])

        # read corresponding part of DB TABLE results_lightgbm_stock
        query = text('SELECT * FROM {} WHERE (trial_lgbm IN :trial_lgbm)'.format(table_name))
        query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
        result_stock = pd.read_sql(query, conn)

    engine.dispose()

    detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')

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

        self.ibes['y_ibes_act'] = (self.ibes['eps1tr12'].shift(-4) - self.ibes['eps1tr12']) / self.ibes['fn_8001'] * self.ibes['fn_5192']
        self.ibes.loc[self.ibes.groupby('identifier').tail(4).index, 'y_ibes_act'] = np.nan     # use ibes ttm for Y

        self.ibes['y_ibes'] = (self.ibes['eps1fd12'] - self.ibes['eps1tr12']) * self.ibes['fn_5192'] / self.ibes['fn_8001']     # use ibes fwd & ttm for Y estimation

        self.ibes = self.label_sector(self.ibes[['identifier', 'period_end', 'y_ibes', 'y_ibes_act','y_ni']]).dropna(how='any')

        return self.ibes

    def label_sector(self, df):
        ''' find sector(6) / industry(2) for each identifier '''

        with engine.connect() as conn:
            icb = pd.read_sql("SELECT icb_sector, identifier FROM dl_value_universe WHERE identifier IS NOT NULL", conn)
        engine.dispose()
        icb['icb_industry'] = icb.dropna(how='any')['icb_sector'].astype(str).str[:2].astype(int)

        return df.merge(icb.drop_duplicates(), on=['identifier'], how='left')   # remove dup due to cross listing

def yoy_to_median(yoy, r_name):
    ''' 2. convert yoy in qcut format to medians with med_train from training set'''

    try:
        bins_df = pd.read_csv('results_bins.csv')
    except:
        with engine.connect() as conn:
            bins_df = pd.read_sql('SELECT * from results_bins', conn)
        engine.dispose()

    # remove duplicated records
    bins_df = bins_df.drop_duplicates(['qcut_q', 'icb_code', 'testing_period', 'y_type', 'combine_industry'], keep='first')

    def to_median(arr, convert):
        ''' convert qcut bins to median of each group '''

        cut_bins = convert['cut_bins'].strip('{}').split(',')   # convert string {1, 2, 3....} to list
        med_test = convert['med_train'].strip('{}').split(',')

        cut_bins[0] = -np.inf  # convert cut_bins into [-inf, ... , inf]
        cut_bins[-1] = np.inf
        cut_bins = [float(x) for x in cut_bins]     # convert string in list to float

        arr_q = pd.cut(arr, bins=cut_bins, labels=False)  # cut original series into 0, 1, .... (bins * n)

        if r_name != 'classification':
            try:
                med_test = [float(x) for x in med_test]
                arr_q = arr_q.replace(range(int(convert['qcut_q'])), med_test).values  # replace 0, 1, ... into median
            except:
                print('fail: ', med_test)

        return arr_q  # return converted Y and median of all groups

    yoy_list = []
    indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
                   501010, 201020, 502030, 401010]
    indi_industry_new = [10, 20, 30, 35, 40, 45, 50, 60, 65]

    for i in tqdm(range(len(bins_df))):   # roll over all cut_bins used by LightGBM -> convert to median


        if bins_df.iloc[i]['icb_code'] == 999999:   # represent miscellaneous model
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (~yoy['icb_sector'].isin(indi_models))]

        elif bins_df.iloc[i]['icb_code'] in (indi_models + indi_industry_new):
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (yoy['icb_sector'] == bins_df.iloc[i]['icb_code'])]

        elif bins_df.iloc[i]['icb_code'] in [0, 1]:  # when using general model
            part_yoy = yoy.loc[yoy['period_end'] == bins_df.iloc[i]['testing_period']]

        else:
            print('fail: ', bins_df.iloc[i]['icb_code'])
            exit(1)

        # qcut (and convert to median if applicable) for y_ibes, y_ni, y_ibes_act
        part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes'], convert=bins_df.iloc[i])
        part_yoy['y_ni_qcut'] = to_median(part_yoy['y_ni'], convert=bins_df.iloc[i])
        part_yoy['y_ibes_act_qcut'] = to_median(part_yoy['y_ibes_act'], convert=bins_df.iloc[i])

        yoy_list.append(part_yoy)

    return pd.concat(yoy_list, axis=0)

def act_lgbm_ibes(detail_stock, yoy_med):
    ''' combine all prediction together '''

    # convert datetime
    yoy_med = date_type(yoy_med)
    detail_stock['exclude_fwd'] = detail_stock['exclude_fwd'].fillna(False)

    # pivot prediction for lgbm with fwd & lgbm without fwd
    detail_stock = pd.merge(detail_stock.loc[detail_stock['exclude_fwd'] == True],
                            detail_stock.loc[detail_stock['exclude_fwd'] == False],
                            on=['identifier', 'qcut_q', 'icb_code', 'testing_period', 'cv_number'],
                            how='outer', suffixes=('_ex_fwd', '_in_fwd'))

    detail_stock = detail_stock.drop_duplicates(subset=['icb_code','identifier','testing_period','cv_number'], keep='last')

    # use median for cross listing & multiple cross-validation
    detail_stock = detail_stock.groupby(['icb_code','identifier','testing_period']).median()[['pred_ex_fwd','pred_in_fwd']].reset_index(drop=False)

    # merge (stock prediction) with (ibes consensus median)
    yoy_merge=detail_stock.merge(yoy_med, left_on=['identifier','testing_period'], right_on=['identifier','period_end'])

    return yoy_merge

def calc_score(yoy_merge, industry, ibes_act, classify):
    ''' calculate mae for each testing_period, icb_code, (cv_number) '''

    def part_mae(df, ibes_act):
        ''' calculate different mae for groups of sample '''

        dict = {}
        if ibes_act == True:
            dict['ibes'] = mean_absolute_error(df['y_ibes_qcut'], df['y_ibes_act_qcut'])
        else:
            dict['ibes'] = mean_absolute_error(df['y_ibes_qcut'], df['y_ni_qcut'])

        for col in ['_ex_fwd', '_in_fwd']:
            try:
                dict['lgbm{}'.format(col)] = mean_absolute_error(df['pred{}'.format(col)], df['y_ni_qcut'])
            except:
                dict['lgbm{}'.format(col)] = np.nan
                continue
        dict['len'] = len(df)
        return dict

    def part_accu(df, ibes_act):
        ''' calculate different mae for groups of sample '''

        dict = {}
        if ibes_act == True:
            dict['ibes'] = accuracy_score(df['y_ibes_qcut'], df['y_ibes_act_qcut'])
        else:
            dict['ibes'] = accuracy_score(df['y_ibes_qcut'], df['y_ni_qcut'])

        dict['lgbm_ex_fwd'] = accuracy_score(df['pred_ex_fwd'], df['y_ni_qcut'])
        dict['len'] = len(df)
        return dict

    mae = {}
    for p in set(yoy_merge['testing_period']):
        part_p = yoy_merge.loc[yoy_merge['testing_period']==p]

        if industry == False:
            for i in set(yoy_merge['icb_code']):
                part_i = part_p.loc[part_p['icb_code']==i]

                if len(part_i) > 0:    # calculate aggregate mae for all 5 cv groups
                    if classify == True:
                        mae['{}_{}_all'.format(p, i)] = part_accu(part_i, ibes_act)
                    else:
                        mae['{}_{}_all'.format(p, i)] = part_mae(part_i, ibes_act)
                else:
                    print('not available (len = 0)', p, i)
                    continue

        else:
            for i in set(yoy_merge['icb_industry']):
                part_i = part_p.loc[part_p['icb_industry'] == i]

                if len(part_i) > 0:    # calculate aggregate mae for all 5 cv groups
                    if classify == True:
                        mae['{}_{}_all'.format(p, i)] = part_accu(part_i, ibes_act)
                    else:
                        mae['{}_{}_all'.format(p, i)] = part_mae(part_i, ibes_act)
                else:
                    print('not available (len = 0)', p, i)
                    continue

    df = pd.DataFrame(mae).T.reset_index()

    df[['testing_period', 'icb_code', 'cv_number']] = df['index'].str.split('_', expand=True)
    df = df.filter(['icb_code','testing_period', 'cv_number','ibes','lgbm_ex_fwd','lgbm_in_fwd','len'])

    return df

def label_ind_name(df):
    '''label sector name for each icb_code '''

    with engine.connect() as conn:
        ind_name = pd.read_sql('SELECT * FROM industry_group', conn)
    engine.dispose()

    df = df.merge(ind_name, left_on=['icb_code'], right_on=['industry_group_code'], how='left')
    return df.drop(['industry_group_code'], axis=1)

def main(r_name):
    ''' main function: clean ibes + calculate mae '''

    # STEP 1: download Data

    try:  # Download 1: download lightgbm results for stocks
        detail_stock = pd.read_csv('results_lgbm/compare_with_ibes/stock_{}.csv'.format(r_name))
        detail_stock = date_type(detail_stock, date_col='testing_period')
        print('local version run - stock_{}'.format(r_name))
    except:
        detail_stock = download_add_detail(r_name,'results_lightgbm_stock')
        detail_stock.to_csv('results_lgbm/compare_with_ibes/stock_{}.csv'.format(r_name), index=False)
    exit(0)

    try:    # Download 2: download ibes_data and organize to YoY
        yoy = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv')
        yoy = date_type(yoy)
        print('local version run - ibes_yoy ')
    except:
        yoy = eps_to_yoy().merge_and_calc()
        yoy.to_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv', index=False)

    yoy_med = yoy_to_median(yoy, r_name)                    # STEP2: convert ibes YoY to qcut / median

    yoy_merge = act_lgbm_ibes(detail_stock, yoy_med)        # STEP3: combine lightgbm and ibes results
    yoy_merge.to_csv('## consensus.csv', index=False)
    exit(0)

    df = calc_score(yoy_merge, r_name)                      # STEP4: calculate MAE / accuracy

    # STEP6: save to EXCEL
    ttm = {True:'ibes', False: 'ws'}
    writer = pd.ExcelWriter('results_lgbm/compare_with_ibes/mae_{}.xlsx'.format(r_name))

    df.to_excel(writer, 'all', index=False)         # all results

    df_sector = df.groupby('icb_code').mean().reset_index()     # results: average of sectors
    label_ind_name(df_sector).to_excel(writer, 'by sector', index=False)

    df.groupby('testing_period').mean().reset_index().to_excel(writer, 'by time', index=False)  # results: average of testing_period
    df.mean().to_excel(writer, 'average')

    print('save to file name: mae_{}.xlsx'.format(r_name))
    writer.save()

def combine():
    ''' combine average of different trial to save csv for comparison'''

    os.chdir('results_lgbm/compare_with_ibes')

    com = []
    for root, dirs, files in os.walk(".", topdown=False):
        for f in files:
            if ('mae_' in f) and ('.xlsx' in f):
                s = pd.read_excel(f, 'average', index_col='Unnamed: 0')[0]
                s.name = f.replace('ibes5_mae_ibes_', '').replace('.xlsx','')
                print(s)
                com.append(s)

    pd.concat(com, axis=1).T.to_csv('# mae_compare_all.csv')

if __name__ == "__main__":

    # DB TABLE results_lightgbm column Name -> distinguish training versions {industry:{classify}}
    name_list = {True: {False:'industry'}, False:{True:'classification', False: 'complete fwd'},
                 'no': {False: 'entire'},  'new': {False: 'new industry'}}

    r_name = 'qcut x - new industry'       #  complete fwd (by sector), industry, classification, new industry, entire

    main(r_name)

    combine()



