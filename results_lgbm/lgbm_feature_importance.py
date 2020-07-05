from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import datetime as dt

def download(update=0):
    ''' donwload results from results_feature_importance '''

    if update==0:   # update if newer results is downloaded
        importance = pd.read_csv('results_lgbm/feature_importance/feature_importance.csv')
        print('local version run - importance')

    elif update==1:
        with engine.connect() as conn:
            importance = pd.read_sql('SELECT * FROM results_feature_importance', con=conn)
        engine.dispose()

        importance.to_csv('results_lgbm/feature_importance/feature_importance.csv', index=False)

    return importance

def map_info(importance):
    ''' mapping training details to feature importance based on trial_lgbm'''

    name = 'restart - without fwd'  # labels for training rounds
    trial_lgbm = set(importance['trial_lgbm'])
    # print(trial_lgbm)

    with engine.connect() as conn:
        query = text("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, exclude_fwd "
                     "FROM results_lightgbm WHERE name='{}' AND (trial_lgbm IN :trial_lgbm)".format(name))
        query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
        info = pd.read_sql(query, conn)
    engine.dispose()

    importance_info = importance.merge(info, on=['trial_lgbm'], how='inner')
    # print(importance_info)

    importance_info.to_csv('results_lgbm/feature_importance/feature_importance_info.csv', index=False)

    return importance_info

def group_icb(df):
    ''' groupby icb_code and find sum'''
    # print(df)
    print(df.groupby(['icb_code','importance_type']).sum())

    df.groupby(['importance_type']).mean().T.to_csv('results_lgbm/feature_importance/feature_importance_sum_sum.csv')

    df.groupby(['icb_code','importance_type']).sum().T.to_csv('results_lgbm/feature_importance/feature_importance_sum.csv')


if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    importance = download(1)
    importance_info = map_info(importance)

    # importance_info = pd.read_csv('results_lgbm/feature_importance/feature_importance_info.csv')
    group_icb(importance_info)