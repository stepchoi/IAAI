import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from miscel import write_db

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def new_results_lightgbm():

    try:
        result_all = pd.read_csv('result_all.csv', low_memory=False)
        print('local version run - result_all')

    except:
        print('-----------------> downloading data from DB TABLE results_lightgbm')

        with engine.connect() as conn:
            result_all = pd.read_sql('SELECT * FROM results_lightgbm', conn)
        engine.dispose()

        result_all.to_csv('result_all.csv', index=False)

    # print(result_all)

    rename = pd.read_excel('ratio_calculation.xlsx','DB_name')

    # 1. reset trial_lgbm to sequential list
    mask = result_all['name'].isin(['new qcut x - new industry', 'ibes qcut x - new industry'])
    last_correct_num = result_all.loc[~mask, 'trial_lgbm'].max()
    # print(last_correct_num)

    wrong_num = result_all.loc[mask, 'trial_lgbm'].to_list()
    right_num = [last_correct_num + x + 1 for x in range(len(wrong_num))]
    result_all.loc[mask, 'trial_lgbm'] = result_all.loc[mask, 'trial_lgbm'].replace(wrong_num, right_num)
    # print(len(wrong_num), wrong_num, len(right_num), right_num)

    # 2. rename
    result_all['name'] = result_all['name'].replace(rename['old_name'].to_list(), rename['new_name'].to_list())

    result_all.to_csv('results_lightgbm_new.csv', index=False)
    #
    # write_db(result_all, 'results_lightgbm_new')

    return last_correct_num, wrong_num, right_num

def new_stock():

    # last_correct_num, wrong_num, right_num = new_results_lightgbm()

    # 3. reset stock records number
    try:
        stock = pd.read_csv('results_all_stock.csv', low_memory=False)
        print('local version run - results_all_stock')

    except:
        print('-----------------> downloading data from DB TABLE results_all_stock')

        with engine.connect() as conn:
            stock = pd.read_sql('SELECT * FROM results_lightgbm_stock', conn)
        engine.dispose()

        stock.to_csv('results_all_stock.csv', index=False)

    print(stock.shape)
    exit(0)

    first_wrong_stock_index = stock.loc[stock['trial_lgbm']==240681].index.to_list()[0]
    # print(first_wrong_stock_index)

    stock[first_wrong_stock_index:] = stock[first_wrong_stock_index:].replace(wrong_num, right_num)

    stock.to_csv('results_lightgbm_stock_new.csv', index=False)

    # write_db(stock, 'results_lightgbm_stock_new')

if __name__ == '__main__':

    new_stock()


