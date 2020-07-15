import numpy as np
import pandas as pd
from sqlalchemy import create_engine


db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def new_results_lightgbm():

    try:
        result_all = pd.read_csv('result_all.csv')
        print('local version run - result_all')

    except:
        print('-----------------> downloading data from DB TABLE results_lightgbm')

        with engine.connect() as conn:
            result_all = pd.read_sql('SELECT * FROM results_lightgbm', conn)
        engine.dispose()

        result_all.to_csv('result_all.csv', index=False)

    print(result_all)

    rename = pd.read_excel('preprocess/ratio_calculation.xlsx','DB_name')

    # 1. reset trial_lgbm to sequential list
    last_correct_num = result_all.loc[result_all['name']!='ibes qcut x - new industry', 'trial_lgbm'].to_list()[-1]
    mask = result_all['name']=='ibes qcut x - new industry'

    wrong_num = result_all.loc[mask, 'trial_lgbm'].to_list()
    right_num = [x + last_correct_num + 1 for x in wrong_num]

    result_all.loc[mask, 'trial_lgbm'] = result_all.loc[mask, 'trial_lgbm'].replace(wrong_num, right_num)
    print(result_all.loc[mask, 'trial_lgbm'])

    # 2. rename
    result_all['name'] = result_all['name'].replace(rename['old_name'].to_list(), rename['new_name'].to_list())

def new_results_lightgbm_stock():

    try:
        result_all = pd.read_csv('results_all_stock.csv')
        print('local version run - results_all_stock')

    except:
        print('-----------------> downloading data from DB TABLE results_all_stock')

        with engine.connect() as conn:
            result_all = pd.read_sql('SELECT * FROM results_all_stock', conn)
        engine.dispose()

        result_all.to_csv('results_all_stock.csv', index=False)

    print(result_all)

if __name__ == '__main__':
    new_results_lightgbm_stock()


