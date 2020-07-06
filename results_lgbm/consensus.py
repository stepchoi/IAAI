from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

class eps_to_yoy:
    ''' calculate 1. IBES forward NET INCOME =
                     IBES forward EPS * common share outstanding used to calculate EPS (both at T0)
                  2. YoY = (forward NI - actual TTM NI) / current Market Cap
    '''

    def __init__(self):
        '''read IBES original data for calculation'''

        try:
            self.ibes = pd.read_csv('preprocess/ibes_data.csv', usecols = ['identifier','period_end',
                                                                           'EPS1FD12','EPS1TR12'])
            self.ws = pd.read_csv('preprocess/ws_ibes.csv')
            self.actual = pd.read_csv('preprocess/clean_ratios.csv', usecols = ['identifier', 'period_end','y_ni'])
            print('local version run - ibes / share_osd')
        except:
            with engine.connect() as conn:
                self.ibes = pd.read_sql('SELECT identifier, period_end, eps1fd12, eps1tr12 FROM ibes_data', conn)
                self.ws = pd.read_sql('SELECT identifier, year, frequency_number, fn_18263, fn_8001, fn_5192 as share_osd '
                                        'FROM worldscope_quarter_summary', conn)
                self.actual = pd.read_sql('SELECT identifier, period_end, y_ni FROM clean_ratios', conn)
            engine.dispose()
            # ws.to_csv('preprocess/ws_ibes.csv', index=False)

    def merge_and_calc(self):
        ''' merge worldscope & ibes dataframe and calculate YoY ratios '''

        self.ibes['period_end'] = pd.to_datetime(self.ibes['period_end'], format='%Y-%m-%d')
        self.actual['period_end'] = pd.to_datetime(self.actual['period_end'], format='%Y-%m-%d')
        self.ibes.columns = ['identifier', 'period_end', 'eps1fd12','eps1tr12']

        # map common share outstanding & market cap to ibes estimations
        self.ws = self.label_period_end(self.ws)
        self.ibes = self.ibes.merge(self.ws, on=['identifier', 'period_end'])
        self.ibes = self.ibes.merge(self.actual, on=['identifier', 'period_end'])

        # calculate YoY (Y)
        self.ibes['pred_ratio'] = self.ibes['eps1fd12'] / self.ibes['eps1tr12']
        self.ibes['fwd_ni'] = self.ibes['pred_ratio'] * self.ibes['fn_18263']
        self.ibes['y_ibes'] = (self.ibes['fwd_ni'] - self.ibes['fn_18263']) / self.ibes['fn_8001']

        # ibes['act_ni'] = ibes['fn_18263'].shift(-4)
        # ibes.loc[ibes.groupby('identifier').tail(4).index, 'act_ni'] = np.nan
        # self.ibes.to_csv('#check_ibes_ni.csv', index=False)

        return self.label_sector(self.ibes[['identifier', 'period_end', 'y_ibes','y_ni']]).dropna(how='any')

    def label_period_end(self, df):
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

    def label_sector(self, df):
        ''' find sector for each identifier '''

        with engine.connect() as conn:
            icb = pd.read_sql("SELECT icb_sector, identifier FROM dl_value_universe WHERE identifier IS NOT NULL", conn)
        engine.dispose()

        return df.merge(icb, on=['identifier'])

def yoy_to_median(yoy):
    ''' convert yoy in qcut format to medians with med_train from training set'''

    with engine.connect() as conn:
        bins_df = pd.read_sql("SELECT * FROM results_bins WHERE med_train !='{\"Not applicable\"}'", conn)
    engine.dispose()

    def to_median(arr, convert):
        ''' convert qcut bins to median of each group '''

        cut_bins = convert['cut_bins'].strip('{}').split(',')   # convert string {1, 2, 3....} to list
        med_test = convert['med_train'].strip('{}').split(',')

        cut_bins[0] = -np.inf  # convert cut_bins into [-inf, ... , inf]
        cut_bins[-1] = np.inf
        cut_bins = [float(x) for x in cut_bins]     # convert string in list to float
        med_test = [float(x) for x in med_test]

        arr_q = pd.cut(arr, bins=cut_bins, labels=False)  # cut original series into 0, 1, .... (bins * n)
        arr_new = arr_q.replace(range(int(convert['qcut_q'])), med_test).values  # replace 0, 1, ... into median

        return arr_new  # return converted Y and median of all groups

    yoy_list = []
    for i in tqdm(range(len(bins_df))):   # roll over all cut_bins used by LightGBM -> convert to median

        if bins_df.iloc[i]['icb_code'] == 999999:   # represent miscellaneous model
            indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
                           501010, 201020, 502030, 401010]
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (~yoy['icb_sector'].isin(indi_models))]

        else:
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (yoy['icb_sector'] == bins_df.iloc[i]['icb_code'])]

        part_yoy['y_ibes_qcut'] = to_median(part_yoy['y_ibes'], convert=bins_df.iloc[i])
        part_yoy['y_ni_qcut'] = to_median(part_yoy['y_ni'], convert=bins_df.iloc[i])

        yoy_list.append(part_yoy)

    yoy_ibes_median = pd.concat(yoy_list, axis=0)

    return yoy_ibes_median

def act_lgbm_ibes(yoy_ibes_median, update):
    ''' combine all prediction together '''

    if update !=1 :     # No Update
        detail_stock = pd.read_csv('results_lgbm/compare_with_ibes/ibes_detail_stock.csv')
        print('local version run - detail_stock')

    elif update == 1:   # Update stock specific results from FB
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
        detail_stock.to_csv('results_lgbm/compare_with_ibes/ibes_detail_stock.csv', index=False)

    # convert datetime
    detail_stock['testing_period'] = pd.to_datetime(detail_stock['testing_period'], format='%Y-%m-%d')
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

def calc_mae(yoy_merge):
    ''' calculate mae for each testing_period, icb_code, (cv_number) '''

    def part_mae(df):
        ''' calculate different mae for groups of sample '''

        dict = {}
        dict['ibes'] = mean_absolute_error(df['y_ibes_qcut'], df['y_ni_qcut'])
        dict['lgbm_ex_fwd'] = mean_absolute_error(df['pred_ex_fwd'], df['y_ni_qcut'])
        dict['lgbm_in_fwd'] = mean_absolute_error(df['pred_in_fwd'], df['y_ni_qcut'])
        dict['len'] = len(df)
        return dict

    mae = {}
    for p in set(yoy_merge['testing_period']):
        part_p = yoy_merge.loc[yoy_merge['testing_period']==p]

        for i in set(yoy_merge['icb_code']):
            part_i = part_p.loc[part_p['icb_code']==i]

            try:    # calculate aggregate mae for all 5 cv groups
                mae['{}_{}_all'.format(p, i)] = part_mae(part_i)
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
    df.drop(['industry_group_code'], axis=1).to_csv('results_lgbm/compare_with_ibes/ibes_mae.csv', index=False)

def main(update=0):
    ''' main function: clean ibes + calculate mae '''

    if update == 1:

        try:
            yoy = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv')
            yoy['period_end'] = pd.to_datetime(yoy['period_end'], format='%Y-%m-%d')
            print('local version run - ibes_yoy ')
        except:
            yoy = eps_to_yoy().merge_and_calc()
            yoy.to_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv', index=False)

        yoy_med = yoy_to_median(yoy)       # Update every time for new cut_bins

        yoy_med.to_csv('results_lgbm/compare_with_ibes/ibes_yoy_median.csv', index=False)
        # yoy_med = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy_median.csv')

        yoy_merge = act_lgbm_ibes(yoy_med, update)
        yoy_merge.to_csv('results_lgbm/compare_with_ibes/ibes_yoy_merge.csv', index=False)

    else:
        yoy_merge = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy_merge.csv')

    calc_mae(yoy_merge)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    main(1)

    # df = pd.read_csv('results_lgbm/compare_with_ibes/ibes_detail_stock.csv', usecols=['trial_lgbm', 'exclude_fwd','mae_test'])
    # print(df)
    #
    # from collections import Counter
    # c = Counter(df['exclude_fwd'])
    # print(c)
    #
    # for name, g in df.groupby(['exclude_fwd']):
    #     print('--------------------> exclude_fwd', name)
    #     print( g.describe())