from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import datetime as dt
from miscel import date_type, check_dup

def download_part(r_name):

    try:  # STEP1: download lightgbm results for feature importance
        importance = pd.read_csv('results_lgbm/feature_importance/total_{}.csv'.format(r_name))
        # importance = date_type(importance, date_col='testing_period')
        print('local version run - total_{}'.format(r_name))

    except:

        exit(0)
        importance = download_result_features(r_name)
        importance.to_csv('results_lgbm/feature_importance/total_{}.csv'.format(r_name), index=False)

    # importance = importance.drop(['trial_lgbm', 'qcut_q', 'testing_period', 'cv_number', 'mae_test', 'exclude_fwd', 'y_type'], axis=1)

    return importance

def download_result_features(r_name, table_name='results_lightgbm'):
    ''' 3. download from DB TABLE results_feature_importance '''

    print('----------------> update stock results from DB: {}'.format(r_name))

    with engine.connect() as conn:

        # read DB TABLE results_lightgbm data for given "name"
        result_all = pd.read_sql("SELECT trial_lgbm, x_type, y_type,icb_code,mae_test, qcut_q FROM {} WHERE name='{}'".format(table_name, r_name), conn)
        trial_lgbm = set(result_all['trial_lgbm'])

        # read corresponding part of DB TABLE results_lightgbm_stock
        query = text('SELECT * FROM results_feature_importance WHERE (trial_lgbm IN :trial_lgbm)')
        query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
        result_stock = pd.read_sql(query, conn)

    engine.dispose()

    print(result_all, result_stock)

    return result_stock.merge(result_all, on=['trial_lgbm'], how='inner')

def download_complete_describe(importance_type):

    # read TABLE results_feature_importance
    # feature = pd.read_csv('results_feature_importance_new.csv')
    #
    # info = pd.read_csv('result_lightgbm_new_new.csv')
    # print(feature.shape, info.shape)
    # feature_info = info[['trial_lgbm', 'x_type', 'y_type','icb_code',
    #                      'mae_test', 'qcut_q']].merge(feature, on='trial_lgbm').drop_duplicates()

    feature_info = download_part('ibes_new industry_qcut x -re')

    feature_info[['trial_lgbm', 'icb_code']] = feature_info[['trial_lgbm', 'icb_code']].astype(str)

    feature_info = feature_info.loc[feature_info['importance_type']==importance_type]

    rank_name ={True: 'rank', False: 'org'}

    for by_rank in [True, False]:

        writer = pd.ExcelWriter('results_lgbm/feature_importance/describe_{}_{}.xlsx'.format(r_name, rank_name[by_rank]))

        feature_info['sample_type'] = [len(x) for x in feature_info['icb_code']]
        # print(feature_info['sample_type'])
        # print(feature_info.groupby(['y_type', 'x_type', 'sample_type','qcut_q']).mean()['mae_test'].reset_index())
        # exit(0)

        for name, g in feature_info.groupby(['y_type', 'x_type']):
            # print(name, g.mean())

            # continue
            org_by_type(g, by_rank).to_excel(writer, '_'.join(name))
        # exit(0)
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

    # table_name = 'results_dense'
    r_name = 'ibes'

    # feature = download_part()
    feature = download_complete_describe(importance_type='split')

