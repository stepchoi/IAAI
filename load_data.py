from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
import gc

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

class add_macro():

    def __init__(self):
        try:
            self.ratios = pd.read_csv('preprocess/clean_ratios.csv')
            self.macros = pd.read_csv('preprocess/clean_macros.csv')
            print('local version run - clean_ratios / macros')
        except:
            self.ratios = pd.read_sql('SELECT * FROM clean_ratios', engine)
            self.macros = pd.read_sql('SELECT * FROM clean_macros', engine)
            engine.dispose()

        self.ratios = self.label_nation_sector(self.ratios)

    def label_nation_sector(self, ratios):
        ''' read files mapping icb_code & market '''

        with engine.connect() as conn:
            markets = pd.read_sql("SELECT index_ric, icb_sector, identifier FROM dl_value_universe "
                                  "WHERE identifier IS NOT NULL", conn)
        engine.dispose()

        ratios = pd.merge(ratios, markets, on=['identifier'])
        ratios['market'] = ratios['index_ric'].replace(['0#.CSI300', '0#.N225', '0#.SPX', '0#.HSLI'], ['CH','JP','US','HK'])
        return ratios.drop(['index_ric'], axis=1)

    def map_macros(self):
        ''' map macros to ratios TABLE '''

        with engine.connect():
            mapping = pd.read_sql('SELECT * FROM macro_formula', engine).iloc[:,:3]
        engine.dispose()

        # map worldwide indicators based on period_end
        worldwide_col = mapping.loc[mapping['market'] == 'W', 'symbol'].to_list()
        self.ratios = pd.merge(self.ratios, self.macros[['period_end'] + worldwide_col], on=['period_end'], how='left')

        # map market-specific indicators based on period_end & market
        new_macros = self.macros.set_index(['period_end']).unstack().reset_index() # pivot -> index (period_end, market)
        new_macros.columns = ['symbol', 'period_end', 'values']
        new_macros = pd.merge(new_macros, mapping, on=['symbol'])
        new_macros = new_macros.loc[new_macros['market'] != 'W']

        new_macros = new_macros.pivot_table(index=['period_end','market'], columns='type', values='values')
        self.ratios = pd.merge(self.ratios, new_macros, on=['period_end', 'market'], how='left')

        self.ratios['period_end'] = pd.to_datetime(self.ratios['period_end'])
        return self.ratios

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self):
        ''' split train and testing set
                    -> return dictionary contain (x, y, y without qcut) & cut_bins'''

        try:
            self.main = pd.read_csv('preprocess/main.csv')
            self.main['period_end'] = pd.to_datetime(self.main['period_end'], format='%Y-%m-%d')
            print('local version run - main')
        except:
            self.main = add_macro().map_macros()
            # self.main.to_csv('preprocess/main.csv', index=False)

        # print('check inf: ', np.any(np.isinf(self.main.drop(['identifier', 'period_end', 'icb_sector', 'market'], axis=1).values)))

        # define self objects
        self.sample_set = {}
        self.cut_bins = {}
        self.sector = pd.DataFrame()
        self.train = pd.DataFrame()

    def split_icb(self, icb_code):
        ''' split samples from specific sectors (icb_code) '''

        indi_model_icb = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
                          501010, 201020, 502030, 401010]

        if icb_code in indi_model_icb:
            self.sector = self.main.loc[self.main['icb_sector'] == icb_code]
        else:
            self.sector = self.main.loc[~self.main['icb_sector'].isin(indi_model_icb)]
            print('This is miscellaneous model')
        # print(main.describe().T[['max','min']])

    def split_train_test(self, testing_period, exclude_fwd):
        ''' split training / testing set based on testing period '''

        # 1. split train / test set
        start = testing_period - relativedelta(years=10) # train df = 40 quarters
        self.train = self.sector.loc[(start <= self.sector['period_end']) &
                              (self.sector['period_end'] < testing_period)].reset_index(drop=True)
        self.test = self.sector.loc[self.sector['period_end'] == testing_period].reset_index(drop=True)

        # 2. split x, y for train / test set
        def divide_set(df):
            ''' split x, y from main '''

            if exclude_fwd == False:
                x = df.drop(['identifier', 'period_end', 'icb_sector', 'market', 'y_ni', 'y_rev'], axis=1)
            else:   # remove 2 ratios calculated with ibes consensus data
                x = df.drop(['identifier', 'period_end', 'icb_sector', 'market', 'y_ni', 'y_rev','fwd_ey','fwd_roic'],
                            axis=1)
            self.feature_names = x.columns.to_list()
            # print('check if exclude_fwd should be 46, we have ', x.shape)

            x = x.values
            ni = df['y_ni'].values
            rev = df['y_rev'].values
            return x, ni, rev

        # keep non-qcut y for calculation
        self.sample_set['train_x'], self.sample_set['train_ni_org'], self.sample_set['train_rev_org'] = divide_set(self.train)
        self.sample_set['test_x'], self.sample_set['test_ni_org'], self.sample_set['test_rev_org'] = divide_set(self.test)

    def standardize_x(self):
        ''' tandardize x with train_x fit '''

        scaler = StandardScaler().fit(self.sample_set['train_x'])
        self.sample_set['train_x'] = scaler.transform(self.sample_set['train_x'])
        self.sample_set['test_x'] = scaler.transform(self.sample_set['test_x']) # can work without test set

    def y_qcut(self, qcut_q, median):
        ''' qcut y '''

        def to_median(median):
            ''' convert qcut bins to median of each group '''

            # cut original series into 0, 1, .... (bins * n)
            train_y, cut_bins = pd.qcut(self.sample_set['train_{}_org'.format(i)], q=qcut_q, retbins=True, labels=False)
            test_y = pd.cut(self.sample_set['test_{}_org'.format(i)], bins=cut_bins, labels=False)

            if median == True:
                # calculate median on train_y for each qcut group
                df = pd.DataFrame(np.vstack((self.sample_set['train_{}_org'.format(i)], np.array(train_y)))).T   # concat original series / qcut series
                median = df.groupby([1]).median().sort_index()[0].to_list()     # find median of each group

                # replace 0, 1, ... into median
                train_y = pd.DataFrame(train_y).replace(range(qcut_q), median)[0].values
                test_y = pd.DataFrame(test_y).replace(range(qcut_q), median)[0].values
            else:
                train_y = np.array(train_y)
                test_y = np.array(test_y)
                median = ['Not applicable']

            return train_y, test_y, list(cut_bins), list(median)

        for i in ['ni', 'rev']: # convert Net Income / Revenue as Y separately
            self.cut_bins[i] = {}

            self.sample_set['train_{}'.format(i)], self.sample_set['test_{}'.format(i)], self.cut_bins[i]['cut_bins'], \
            self.cut_bins[i]['med_train'] = to_median()


    def split_valid(self, y_type):
        ''' split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set '''

        gkf = GroupShuffleSplit(n_splits=5).split(self.sample_set['train_x'], self.sample_set['train_{}'.format(y_type)]
                                                  , groups=self.train['identifier'])
        return gkf

    def split_all(self, testing_period, qcut_q, y_type='ni', exclude_fwd=False, median=True):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, exclude_fwd)
        self.standardize_x()
        self.y_qcut(qcut_q, median)
        gkf = self.split_valid(y_type)

        print('sample_set keys: ', self.sample_set.keys())

        return self.sample_set, self.cut_bins, gkf, self.test['identifier'].to_list(), self.feature_names

def count_by_sector(main):
    ''' counter # sample for each sector to decide sector-wide models & miscellaneous model'''

    with engine.connect() as conn:
        name = pd.read_sql('SELECT * FROM industry_group', conn)
    engine.dispose()
    name['industry_group_code'] = name['industry_group_code'].apply(pd.to_numeric, errors='coerce')
    name = name.dropna(how='any').set_index(['industry_group_code'])

    icb = pd.DataFrame.from_dict(Counter(main['icb_sector']), orient='index')
    icb.merge(name, left_index=True, right_index=True).to_csv('count_by_sector.csv')

def count_by_missing(main):
    pd.DataFrame(main.isnull().sum()).to_csv('count_by_missing.csv')

def count_by_year(main):
    main['year'] = main['period_end'].dt.year
    pd.DataFrame.from_dict(Counter(main['year']), orient='index').to_csv('count_by_year.csv')

if __name__ == '__main__':

    # these are parameters used to load_data
    icb_code = 999999
    exclude_fwd = True
    testing_period = dt.datetime(2013,3,31)
    qcut_q = 10
    valid_method = 'shuffle'
    valid_no = 10

    data = load_data()
    data.split_icb(icb_code)
    sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, qcut_q, 'ni', exclude_fwd)

    # check shape of sample sets (x + y + y_org) * (train + valid + test)
    print(cut_bins)


    for k in sample_set.keys():
        print(k, sample_set[k].shape, type(sample_set[k]))

    # y1 = pd.DataFrame(np.vstack((sample_set['train_y'], sample_set['train_y_org'])).T)
    # y1.to_csv('#df_check_y1.csv')

