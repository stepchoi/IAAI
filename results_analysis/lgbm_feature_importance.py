from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import datetime as dt
from miscel import date_type, check_dup

def download_result_features(r_name=[]):
    ''' 3. download from DB TABLE results_feature_importance '''

    print('----------------> update stock results from DB {}'.format(r_name))

    with engine.connect() as conn:

        # read DB TABLE results_lightgbm data for given "name"
        if r_name == []:
            result_all = pd.read_sql("SELECT trial_lgbm, x_type, y_type,icb_code FROM results_lightgbm "
                                     "WHERE y_type ='ibes", conn)
        else:
            query_1 = text('SELECT trial_lgbm, x_type, y_type,icb_code FROM results_lightgbm WHERE (name IN :r_name)')
            query_1 = query_1.bindparams(r_name=tuple(r_name))
            result_all = pd.read_sql(query_1, conn)

        print('result_all: ', result_all.shape)
        trial_lgbm = set(result_all['trial_lgbm'])

        # read corresponding part of DB TABLE results_lightgbm_stock
        query = text('SELECT * FROM results_feature_importance WHERE (trial_lgbm IN :trial_lgbm)')
        query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
        result_stock = pd.read_sql(query, conn)
        print('result_stock: ', result_stock.shape)

    engine.dispose()

    return result_stock.merge(result_all, on=['trial_lgbm'], how='inner')

class download:
    ''' download data from DB / read local csv about results_feature_importance'''

    def __init__(self, r_name):
        if r_name == ['all']:
            self.download_complete = self.download_complete()
        else:
            self.download = self.download_part(r_name)

    def download_part(self, r_name):
        try:  # STEP1: download lightgbm results for feature importance
            importance = pd.read_csv('results_analysis/feature_importance/total_{}.csv'.format(r_name))
            print('local version run - total_{}'.format(r_name))
        except:
            importance = download_result_features(r_name)
            importance.to_csv('results_analysis/feature_importance/total_{}.csv'.format(r_name), index=False)
        return importance

    def download_complete(self):

        try:  # STEP1: download lightgbm results for feature importance
            importance = pd.read_csv('results_analysis/feature_importance/total_ibes.csv')
            print('local version run - total_ibes')
        except:
            importance = download_result_features()
            importance.to_csv('results_analysis/feature_importance/total_ibes.csv', index=False)
        return importance

    def finish(self):
        x_col = list(set(self.download.columns.to_list()) - {'qcut_q', 'testing_period', 'cv_number', 'mae_test', 'exclude_fwd'})
        return self.download.filter(x_col)

def org_describe(feature_info, importance_type='split', tname=''):

    print(feature_info)

    if 'entire' in r_name:  # for entire
        print('------ convert entire ------')
        feature_info.loc[feature_info['icb_code'] == 1, 'x_type'] = 'fwdepsqcut-industry_code'
        feature_info.loc[feature_info['icb_code'] == 2, 'x_type'] = 'fwdepsqcut-sector_code'
        feature_info['icb_code'] = 0

    x_col = ['gross_margin', 'cap_adequacy_ratio', 'ebitda_to_ev', 'cash_ratio', 'interest_to_earnings', 'div_payout', 'capex_to_dda',
       'fa_turnover', 'inv_turnover', 'earnings_yield', 'ni_to_cfo', 'ca_turnover', 'roe', 'roic', 'debt_to_asset', 'pretax_margin', 'ni_ts01', 'sales_ts01', 'pretax_margin_ts01', 'cfps_ts01', 'ni_ts13', 'sales_ts13', 'pretax_margin_ts13', 'cfps_ts13', 'ni_ts35',
       'sales_ts35', 'pretax_margin_ts35', 'cfps_ts35', 'fwd_ey', 'fwd_roic',
       'stock_return_1qa', 'stock_return_3qb', 'dollar_index', 'crudoil',
       'usinter3', 'usfrbpim', 'usrettotb', 'ushouse.o', 'usgdp...d',
       'uscnper.d', 'cpi', 'gdp', 'ipi', 'reer', 'index', 'interest_rate_10y',
       'interest_rate_3m', 'unemployment', 'eps_ts01',
       'eps_ts13', 'eps_ts35', 'ibes_qcut_as_x', 'icb_sector_x',
       'icb_industry_x']

    feature_info = feature_info.filter(['x_type', 'trial_lgbm', 'icb_code', 'importance_type', 'y_type'] + x_col)

    feature_info['x_type'] = feature_info['x_type'].fillna('fwdepsqcut')

    feature_info[['trial_lgbm', 'icb_code']] = feature_info[['trial_lgbm', 'icb_code']].astype(str)

    feature_info = feature_info.loc[feature_info['importance_type']==importance_type]

    rank_name ={True: 'rank', False: 'org'}

    for by_rank in [True]:  # default calculate ranking importance

        writer = pd.ExcelWriter('results_analysis/feature_importance/describe_ibes_{}{}.xlsx'.format(rank_name[by_rank], tname))

        for name, g in feature_info.groupby(['y_type', 'x_type']):
            org_by_type(g, by_rank).to_excel(writer, '_'.join(name))

        writer.save()

def org_by_type(df_part, by_rank):

    num_col = df_part.select_dtypes(include=[np.number]).columns.to_list()

    if by_rank == True:
        df_part[num_col] = df_part[num_col].rank(axis=1, ascending=False)
        df_rank = df_part.groupby('icb_code').median().T
        df_rank['all'] = df_rank.sum(1).replace(0, np.nan)
        return df_rank.sort_values('all', ascending=True)
    else:
        df_sum = df_part.groupby(['icb_code']).median().T
        df_sum['all'] = df_sum.mean(1).replace(0, np.nan)
        return df_sum.sort_values('all', ascending=False)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    r_name = ['ibes_new industry_only ws -indi space']

    feature = download(r_name).finish()
    # feature = pd.read_csv('201030 feature importance.csv')
    org_describe(feature, importance_type ='split', tname=r_name)
    # feature = download_complete_describe(r_name, importance_type='split')
