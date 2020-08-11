from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter
from results_analysis.lgbm_consensus import eps_to_yoy
from miscel import date_type
import gc
from miscel import check_dup
from preprocess.ratios import full_period

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

indi_sectors = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
                501010, 201020, 502030, 401010]

best_col = ['earnings_yield', 'fa_turnover', 'stock_return_1qa', 'gross_margin', 'stock_return_3qb', 'div_payout',
            'sales_ts01', 'inv_turnover', 'ebitda_to_ev', 'capex_to_dda', 'ni_ts01', 'interest_to_earnings',
            'ni_to_cfo', 'pretax_margin_ts01', 'cash_ratio'] # top 15 most important features for aggregate model

idd = 'C156E0340'
def check_id(df, id=idd):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df.loc[df['identifier'] ==id, ['period_end', 'y_ibes']].sort_values(['period_end']))
    exit(0)

def filter_sp_only(df):
    ''' select members from S&P500 index for training '''

    with engine.connect() as conn:
        sp = pd.read_sql('SELECT * FROM dl_value_universe', conn)       # read index-ticker-id mapping from DB
    engine.dispose()

    sp = sp.loc[sp['index_ric']=='0#.SPX', 'identifier'].to_list()      # select SPX members

    print(df.shape, len(set(df['identifier'])))
    df = df.loc[df['identifier'].isin(sp)]                              # filter companies
    print(df.shape, len(set(df['identifier'])))

    return df

class add_macro:

    def __init__(self, macro_monthly=True):
        try:
            self.ratios = pd.read_csv('preprocess/clean_ratios.csv')
            self.macros = pd.read_csv('preprocess/clean_macros.csv')
            self.new_macros = pd.read_csv('preprocess/clean_macros_new.csv')
            print('local version run - clean_ratios / macros')
        except:
            print('-----------------> download data from DB: clean_ratios/clean_macros')
            with engine.connect() as conn:
                self.ratios = pd.read_sql('SELECT * FROM clean_ratios', conn)
                self.macros = pd.read_sql('SELECT * FROM clean_macros', conn)
                self.new_macros = pd.read_sql('SELECT * FROM clean_macros_new', conn)
            engine.dispose()

        self.macros = date_type(self.macros)    # convert to date

        self.new_macros = date_type(self.new_macros)
        self.ratios = date_type(self.ratios)
        self.macros = self.macros.loc[self.macros['period_end'] >= dt.datetime(1997,12,31)] # filter records after 1998

        if macro_monthly == True:
            non_replace_col = list(set(self.macros.columns.to_list()) - set(self.new_macros.columns.to_list()))
            self.macros = self.new_macros.merge(self.macros[['period_end'] + non_replace_col], on='period_end', how='left')
            print('update using monthly macros')

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
        self.ratios = self.ratios.sort_values('market').drop_duplicates(['identifier', 'period_end'], keep='first')  # for cross listing (CH + HK) use macro for CH
        self.ratios['period_end'] = pd.to_datetime(self.ratios['period_end'])

        return self.ratios

class load_data:
    ''' main function:
        1. split train + valid + test -> sample set
        2. convert x with standardization, y with qcut '''

    def __init__(self, macro_monthly=True, sp_only=False):
        ''' split train and testing set
                    -> return dictionary contain (x, y, y without qcut) & cut_bins'''

        self.main = add_macro(macro_monthly).map_macros()
        self.consensus = eps_to_yoy().merge_and_calc().filter(['identifier','period_end','y_consensus'])

        self.main = date_type(self.main)
        self.main.columns = [x.lower() for x in self.main]

        self.consensus = date_type(self.consensus)
        self.consensus.columns = ['identifier', 'period_end', 'ibes_qcut_as_x']
        self.main = self.main.merge(self.consensus, on =['identifier','period_end'], how='left')

        self.main = self.main.dropna(subset=['icb_sector'])
        self.main['icb_industry'] = self.main['icb_sector'].astype(str).str[:2].astype(int)
        print('main_consensus: ', self.main.shape)

        if sp_only==True:
            self.main = filter_sp_only(self.main)

        # print('check inf: ', np.any(np.isinf(self.main.drop(['identifier', 'period_end', 'icb_sector', 'market'], axis=1).values)))

        # define self objects
        self.sample_set = {}
        self.cut_bins = {}
        self.sector = pd.DataFrame()
        self.train = pd.DataFrame()

    def split_sector(self, icb_code):
        ''' split samples from specific sectors (icb_code) '''

        if icb_code in indi_sectors:
            self.sector = self.main.loc[self.main['icb_sector'] == icb_code]
        else:
            self.sector = self.main.loc[~self.main['icb_sector'].isin(indi_sectors)]
            print('This is miscellaneous model')

    def split_industry(self, icb_industry, combine_ind=True):

        if icb_industry < 10:   # entire sampling
            self.split_entire(icb_industry)
        elif icb_industry > 100:    # per sector
            self.split_sector(icb_industry)
        else:       # per industry
            if combine_ind == True:
                self.main['icb_industry'] = self.main['icb_industry'].replace([10, 15, 50, 55], [11, 11, 51, 51])   # use 11 to represent combined industry (10+15)

            self.sector = self.main.loc[self.main['icb_industry'] == icb_industry]

    def split_entire(self, add_ind_code):
        ''' train on all sample, add_ind_code = True means adding industry_code(2) as x '''

        if add_ind_code == 1:
            self.main['icb_industry_x'] = self.main['icb_industry']
        elif add_ind_code == 2:
            self.main['icb_sector_x'] = self.main['icb_sector']
        elif add_ind_code == 0:
            pass
        else:
            print('wrong add_ind_code !!!!!!! ')
            exit(1)

        self.sector = self.main

    def split_train_test(self, testing_period, exclude_fwd, ibes_qcut_as_x, y_type, exclude_stock, filter_best_col):
        ''' split training / testing set based on testing period '''

        # 1. split train / test set
        start = testing_period - relativedelta(years=10)    # train df = 40 quarters

        self.sector = full_period(date_type(self.sector))
        self.sector = self.sector.loc[~self.sector['ibes_qcut_as_x'].isnull()]

        # self.sector['lookback_y_{}'.format(y_type)] = self.sector['y_{}'.format(y_type)].shift(20)
        # self.sector.iloc[self.sector.groupby('identifier').head(20),'lookback_y_{}'.format(y_type)]
        # full_hist_comp = self.sector.loc[self.sector['period_end']==start, 'identifier'].to_list()
        # self.sector = self.sector[self.sector['period_end'].isin(full_hist_comp)]
        # print(start, len(full_hist_comp), full_hist_comp)

        self.sector = self.sector.dropna(subset=['y_{}'.format(y_type)])    # remove companies with NaN y_ibes
        self.train = self.sector.loc[(start <= self.sector['period_end']) &
                              (self.sector['period_end'] < testing_period)].reset_index(drop=True)
        self.test = self.sector.loc[self.sector['period_end'] == testing_period].reset_index(drop=True)

        print('test_df: ', self.test.shape)

        # 2. split x, y for train / test set
        def divide_set(df, ibes_qcut_as_x):
            ''' split x, y from main '''

            y_col = [x for x in df.columns if x[:2]=='y_']
            fwd_col = ['fwd_ey','fwd_roic']
            fwd_eps_col = ['eps_ts01', 'eps_ts13', 'eps_ts35']
            ws_ni_col = ['ni_ts01','ni_ts13','ni_ts35']
            id_col = ['identifier', 'period_end', 'icb_sector', 'market', 'icb_industry']

            if exclude_fwd == False:
                x = df.drop(id_col + y_col + ws_ni_col , axis=1)
                if ibes_qcut_as_x == False:
                    x = x.drop(['ibes_qcut_as_x'], axis=1)
            else:   # remove 2 ratios calculated with ibes consensus data
                x = df.drop(id_col + y_col + fwd_eps_col + fwd_col, axis=1)
                if ibes_qcut_as_x == False:
                    x = x.drop(['ibes_qcut_as_x'], axis=1)

            print(x)
            if exclude_stock == True:   # for trial without stock_return_1qa data (Lightgbm)
                x = x.drop(['stock_return_1qa'], axis=1)

            if filter_best_col == True:       # for trial with only top N important features (dense2)
                x = x.filter(best_col)
                print('------> Using top {} most important feature: '.format(len(best_col)), best_col)

            self.feature_names = x.columns.to_list()
            # print('check if exclude_fwd should be 46, we have ', x.shape)

            # print('x_col: ', x.columns)
            x = x.values
            y = {}
            for col in y_col:
                y[col[2:]] = df[col].values

            return x, y

        # keep non-qcut y for calculation
        self.sample_set['train_x'], self.sample_set['train_y'] = divide_set(self.train, ibes_qcut_as_x)
        self.sample_set['test_x'], self.sample_set['test_y'] = divide_set(self.test, ibes_qcut_as_x)

    def standardize_x(self):
        ''' tandardize x with train_x fit '''

        scaler = StandardScaler().fit(self.sample_set['train_x'][:,:-1])
        self.sample_set['train_x'][:,:-1] = scaler.transform(self.sample_set['train_x'][:,:-1])
        self.sample_set['test_x'][:,:-1] = scaler.transform(self.sample_set['test_x'][:,:-1]) # can work without test set

    def y_qcut(self, qcut_q, use_median, y_type, ibes_qcut_as_x):
        ''' qcut y '''

        def to_median(use_median):
            ''' convert qcut bins to median of each group '''

            # cut original series into 0, 1, .... (bins * n)
            train_y, cut_bins = pd.qcut(self.sample_set['train_y'][y_type], q=qcut_q, retbins=True, labels=False)
            cut_bins[0], cut_bins[-1] = [-np.inf, np.inf]

            test_y = pd.cut(self.sample_set['test_y'][y_type], bins=cut_bins, labels=False)

            if ibes_qcut_as_x == True:
                self.sample_set['train_x'][:,-1] = pd.cut(self.sample_set['train_x'][:,-1], bins=cut_bins, labels=False)
                self.sample_set['test_x'][:,-1] = pd.cut(self.sample_set['test_x'][:,-1], bins=cut_bins, labels=False)

            if use_median == True:
                # calculate median on train_y for each qcut group
                df = pd.DataFrame(np.vstack((self.sample_set['train_y'][y_type], np.array(train_y)))).T   # concat original series / qcut series
                median = df.groupby([1]).median().sort_index()[0].to_list()     # find median of each group

                # replace 0, 1, ... into median
                train_y = pd.DataFrame(train_y).replace(range(qcut_q), median)[0].values
                test_y = pd.DataFrame(test_y).replace(range(qcut_q), median)[0].values
            else:
                train_y = np.array(train_y)
                test_y = np.array(test_y)
                median = ['Not applicable']

            return train_y, test_y, list(cut_bins), list(median)

        self.cut_bins = {}
        self.sample_set['train_y'], self.sample_set['test_y'], self.cut_bins['cut_bins'], self.cut_bins['med_train'] = to_median(use_median)


    def split_valid(self, testing_period, chron_valid):
        ''' split 5-Fold cross validation testing set -> 5 tuple contain lists for Training / Validation set '''

        if chron_valid == False:    # split validation by stocks
            gkf = GroupShuffleSplit(n_splits=5).split(self.sample_set['train_x'],
                                                      self.sample_set['train_y'],
                                                      groups=self.train['identifier'])

        if chron_valid == True:     # split validation set by chronological order
            valid_period = testing_period - 8 * relativedelta(months=3)   # using last 2 year samples as valid set
            test_index = self.train.loc[self.train['period_end'] >= valid_period].index.to_list()
            train_index = self.train.loc[self.train['period_end'] < valid_period].index.to_list()
            gkf = [(train_index, test_index)]

        return gkf

    def split_all(self, testing_period, qcut_q, y_type='ni', exclude_fwd=False, use_median=True, chron_valid=False,
                  ibes_qcut_as_x=False, exclude_stock=False, filter_best_col=False):
        ''' work through cleansing process '''

        self.split_train_test(testing_period, exclude_fwd, ibes_qcut_as_x, y_type, exclude_stock, filter_best_col)
        self.standardize_x()
        self.y_qcut(qcut_q, use_median, y_type, ibes_qcut_as_x)
        gkf = self.split_valid(testing_period, chron_valid)

        # print('sample_set keys: ', self.sample_set.keys())

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

    use_median = True
    chron_valid = False

    # these are parameters used to load_data
    icb_code = 2
    testing_period = dt.datetime(2018,3,31)
    qcut_q = 10
    y_type = 'ibes'

    exclude_fwd = True
    ibes_qcut_as_x = True
    macro_monthly = True

    data = load_data(sp_only=True)
    # data.split_sector(icb_code)
    data.split_industry(icb_code, combine_ind=True)

    sample_set, cut_bins, cv, test_id, feature_names = data.split_all(testing_period, qcut_q,
                                                                      y_type=y_type,
                                                                      exclude_fwd=exclude_fwd,
                                                                      use_median=use_median,
                                                                      chron_valid=chron_valid,
                                                                      ibes_qcut_as_x=ibes_qcut_as_x,
                                                                      exclude_stock=True)

    print(sorted(feature_names))

    print('test_id: ', len(test_id))
    pd.DataFrame(test_id, columns=['lgbm_id']).to_csv('lgbm_id.csv', index=False)
    print(pd.DataFrame(test_id, columns=['lgbm_id']))



    for train_index, test_index in cv:
        print(len(train_index), len(test_index))

    exit(0)

    # check shape of sample sets (x + y + y_org) * (train + valid + test)
    print(cut_bins)


    for k in sample_set.keys():
        print(k, sample_set[k].shape, type(sample_set[k]))

    # y1 = pd.DataFrame(np.vstack((sample_set['train_y'], sample_set['train_y_org'])).T)
    # y1.to_csv('#df_check_y1.csv')

