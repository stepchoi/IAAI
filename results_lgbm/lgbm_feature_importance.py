from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import datetime as dt
from miscel import date_type, check_dup

def download_result_features(r_name, table_name='results_lightgbm'):
    ''' 3. download from DB TABLE results_lightgbm_stock '''

    print('----------------> update stock results from DB ')

    with engine.connect() as conn:

        # read DB TABLE results_lightgbm data for given "name"
        result_all = pd.read_sql("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, exclude_fwd "
                     "FROM {} WHERE name='{}'".format(table_name, r_name), conn)
        trial_lgbm = set(result_all['trial_lgbm'])

        # read corresponding part of DB TABLE results_lightgbm_stock
        query = text('SELECT * FROM results_feature_importance WHERE (trial_lgbm IN :trial_lgbm)')
        query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
        result_stock = pd.read_sql(query, conn)

    engine.dispose()

    return result_stock.merge(result_all, on=['trial_lgbm'], how='inner')


def group_icb(df):
    ''' groupby icb_code and find sum'''

    # df = df.loc[df['exclude_fwd']==False]
    df = df.drop(['trial_lgbm', 'qcut_q', 'testing_period', 'cv_number', 'mae_test', 'exclude_fwd'], axis=1)

    def org_by_type(importance_type):
        df_type = df.loc[(df['importance_type']==importance_type)].groupby(['icb_code']).mean().T
        df_type['all'] = df_type.sum(1)
        print(df_type)
        return df_type.sort_values('all', ascending=False)

    writer = pd.ExcelWriter('results_lgbm/feature_importance/{}_describe.xlsx'.format(r_name))

    for t in ['gain', 'split']:
        org_by_type(t).to_excel(writer, t)

    writer.save()

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    r_name = 'ibes eps ts - new industry'
    # table_name = 'results_dense'

    try:    # STEP1: download lightgbm results for feature importance
        importance = pd.read_csv('results_lgbm/feature_importance/{}_total.csv'.format(r_name))
        importance = date_type(importance, date_col='testing_period')
        print('local version run - total_{}'.format(r_name))
    except:
        importance = download_result_features(r_name)
        importance.to_csv('results_lgbm/feature_importance/{}_total.csv'.format(r_name), index=False)

    group_icb(importance)