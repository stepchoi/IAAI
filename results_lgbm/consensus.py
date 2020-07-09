from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
from preprocess.ratios import full_period, worldscope
from miscel import date_type

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

        return df.merge(icb, on=['identifier'])

def yoy_to_median(yoy, industry, classify):
    ''' 2. convert yoy in qcut format to medians with med_train from training set'''

    with engine.connect() as conn:
        if industry == True:
            bins_df = pd.read_sql("SELECT * FROM results_bins WHERE med_train !='{\"Not applicable\"}' AND icb_code < 100", conn)
        else:
            bins_df = pd.read_sql("SELECT * FROM results_bins WHERE med_train !='{\"Not applicable\"}' AND icb_code > 100", conn)
    engine.dispose()

    def to_median(arr, convert, classify):
        ''' convert qcut bins to median of each group '''

        cut_bins = convert['cut_bins'].strip('{}').split(',')   # convert string {1, 2, 3....} to list
        med_test = convert['med_train'].strip('{}').split(',')

        cut_bins[0] = -np.inf  # convert cut_bins into [-inf, ... , inf]
        cut_bins[-1] = np.inf
        cut_bins = [float(x) for x in cut_bins]     # convert string in list to float
        med_test = [float(x) for x in med_test]

        arr_q = pd.cut(arr, bins=cut_bins, labels=False)  # cut original series into 0, 1, .... (bins * n)

        if classify == False:
            arr_q = arr_q.replace(range(int(convert['qcut_q'])), med_test).values  # replace 0, 1, ... into median

        return arr_q  # return converted Y and median of all groups

    yoy_list = []
    for i in tqdm(range(len(bins_df))):   # roll over all cut_bins used by LightGBM -> convert to median

        if industry == False:
            if bins_df.iloc[i]['icb_code'] == 999999:   # represent miscellaneous model
                indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
                               501010, 201020, 502030, 401010]
                part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                                   (~yoy['icb_sector'].isin(indi_models))]

            else:
                part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                                   (yoy['icb_sector'] == bins_df.iloc[i]['icb_code'])]
        else:
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (yoy['icb_industry'] == bins_df.iloc[i]['icb_code'])]

        part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes'], convert=bins_df.iloc[i], classify=classify)
        part_yoy['y_ni_qcut'] = to_median(part_yoy['y_ni'], convert=bins_df.iloc[i], classify=classify)
        part_yoy['y_ibes_act_qcut'] = to_median(part_yoy['y_ibes_act'], convert=bins_df.iloc[i], classify=classify)

        yoy_list.append(part_yoy)

    yoy_ibes_median = pd.concat(yoy_list, axis=0)

    return yoy_ibes_median

def download_result_stock():
    ''' 3. download from DB TABLE results_lightgbm_stock '''

    print('----------------> update stock results from DB ')

    with engine.connect() as conn:
        result_stock = pd.read_sql('SELECT * FROM results_lightgbm_stock', conn)
        trial_lgbm = set(result_stock['trial_lgbm'])

        query = text("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, exclude_fwd "
                     "FROM results_lightgbm WHERE name='restart - without fwd' AND (trial_lgbm IN :trial_lgbm)")
        query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
        result_all = pd.read_sql(query, conn)

    engine.dispose()

    detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')

    return detail_stock

def act_lgbm_ibes(detail_stock, yoy_ibes_median):
    ''' 4. combine all prediction together '''

    # convert datetime
    yoy_ibes_median['period_end'] = pd.to_datetime(yoy_ibes_median['period_end'], format='%Y-%m-%d')
    detail_stock['exclude_fwd'] = detail_stock['exclude_fwd'].fillna(False)

    # pivot prediction for lgbm with fwd & lgbm without fwd
    detail_stock = pd.merge(detail_stock.loc[detail_stock['exclude_fwd'] == True],
                            detail_stock.loc[detail_stock['exclude_fwd'] == False],
                            on=['identifier', 'qcut_q', 'icb_code', 'testing_period', 'cv_number'], how='outer')
    detail_stock = detail_stock.filter(['identifier', 'qcut_q', 'icb_code', 'testing_period', 'cv_number',
                                        'pred_x', 'pred_y'])
    detail_stock.columns = detail_stock.columns.to_list()[:-2] + ['pred_ex_fwd', 'pred_in_fwd']

    # merge (stock prediction) with (ibes consensus median)
    yoy_merge = detail_stock.merge(yoy_ibes_median, left_on=['identifier', 'testing_period'],
                                      right_on=['identifier', 'period_end'])

    return yoy_merge

def calc_score(yoy_merge, industry, ibes_act, classify):
    ''' 5. calculate mae for each testing_period, icb_code, (cv_number) '''

    def part_mae(df, ibes_act):
        ''' calculate different mae for groups of sample '''

        dict = {}
        if ibes_act == True:
            dict['ibes'] = mean_absolute_error(df['y_ibes_qcut'], df['y_ibes_act_qcut'])
        else:
            dict['ibes'] = mean_absolute_error(df['y_ibes_qcut'], df['y_ni_qcut'])

        dict['lgbm_ex_fwd'] = mean_absolute_error(df['pred_ex_fwd'], df['y_ni_qcut'])
        dict['lgbm_in_fwd'] = mean_absolute_error(df['pred_in_fwd'], df['y_ni_qcut'])
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
        dict['lgbm_in_fwd'] = accuracy_score(df['pred_in_fwd'], df['y_ni_qcut'])
        dict['len'] = len(df)
        return dict

    mae = {}
    for p in set(yoy_merge['testing_period']):
        part_p = yoy_merge.loc[yoy_merge['testing_period']==p]

        if industry == False:
            for i in set(yoy_merge['icb_code']):
                part_i = part_p.loc[part_p['icb_code']==i]

                try:    # calculate aggregate mae for all 5 cv groups
                    if classify == True:
                        mae['{}_{}_all'.format(p, i)] = part_accu(part_i, ibes_act)
                    else:
                        mae['{}_{}_all'.format(p, i)] = part_mae(part_i, ibes_act)
                except:
                    print('not available', p, i)
                    continue

        else:
            for i in set(yoy_merge['icb_industry']):
                part_i = part_p.loc[part_p['icb_industry'] == i]

                try:  # calculate aggregate mae for all 5 cv groups
                    if classify == True:
                        mae['{}_{}_all'.format(p, i)] = part_accu(part_i, ibes_act)
                    else:
                        mae['{}_{}_all'.format(p, i)] = part_mae(part_i, ibes_act)
                except:
                    print('not available', p, i)
                    continue

    df = pd.DataFrame(mae).T.reset_index()

    df[['testing_period', 'icb_code', 'cv_number']] = df['index'].str.split('_', expand=True)
    df = df.filter(['icb_code','testing_period', 'cv_number','ibes','lgbm_ex_fwd','lgbm_in_fwd','len'])

    # label sector name for each icb_code
    with engine.connect() as conn:
        ind_name = pd.read_sql('SELECT * FROM industry_group', conn)
    engine.dispose()

    df = df.merge(ind_name, left_on=['icb_code'], right_on=['industry_group_code'], how='left')
    return df.drop(['industry_group_code'], axis=1)

def main(industry=False, ibes_act = False, classify=False):
    ''' main function: clean ibes + calculate mae '''

    try:    # STEP1: download ibes_data and organize to YoY
        yoy = pd.read_csv('results_lgbm/compare_with_ibes/ibes1_yoy.csv')
        yoy['period_end'] = pd.to_datetime(yoy['period_end'], format='%Y-%m-%d')
        print('local version run - 1. ibes_yoy ')
    except:
        yoy = eps_to_yoy().merge_and_calc()
        yoy.to_csv('results_lgbm/compare_with_ibes/ibes1_yoy.csv', index=False)

    try:    # STEP2: convert ibes YoY to qcut / median
        yoy_med = pd.read_csv('results_lgbm/compare_with_ibes/ibes2_yoy_median.csv')
        print('local version run - 2. ibes_median ')
    except:
        yoy_med = yoy_to_median(yoy, industry, classify)  # Update every time for new cut_bins
        yoy_med.to_csv('results_lgbm/compare_with_ibes/ibes2_yoy_median.csv', index=False)

    try:    # STEP3: download lightgbm results for stocks
        detail_stock = pd.read_csv('results_lgbm/compare_with_ibes/ibes3_detail_stock.csv')
        detail_stock['testing_period'] = pd.to_datetime(detail_stock['testing_period'], format='%Y-%m-%d')
        print('local version run - 3. detail_stock')
    except:
        detail_stock = download_result_stock()
        detail_stock.to_csv('results_lgbm/compare_with_ibes/ibes3_detail_stock.csv', index=False)

    try:    # STEP4: combine lightgbm and ibes results
        yoy_merge = pd.read_csv('results_lgbm/compare_with_ibes/ibes4_yoy_merge.csv')   # delete this file for stock results update
        print('local version run - 4. yoy_merge')
    except:
        yoy_merge = act_lgbm_ibes(detail_stock, yoy_med)
        yoy_merge.to_csv('results_lgbm/compare_with_ibes/ibes4_yoy_merge.csv', index=False)

    df = calc_score(yoy_merge, industry, ibes_act, classify)  # STEP5: calculate MAE

    name = {True:{True:'ibesttm_industry', False:'ibesttm_sector'}, False: {True:'wsttm_industry', False:'wsttm_sector'}}
    print('save to file name: ibes5_mae_{}'.format(name[ibes_act][industry]))
    df.to_csv('results_lgbm/compare_with_ibes/ibes5_mae_{}.csv'.format(name[ibes_act][industry]), index=False)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    main(industry=False, ibes_act=False, classify=True)  # change to csv name
    exit(0)



