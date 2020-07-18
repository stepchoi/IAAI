import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from miscel import write_db, check_dup

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
    try:
        rename = pd.read_excel('preprocess/ratio_calculation.xlsx','DB_name')
    except:
        rename = pd.read_excel('ratio_calculation.xlsx','DB_name')

    # 1. reset trial_lgbm to sequential list
    mask = result_all['name'].isin(['new qcut x - new industry', 'ibes qcut x - new industry'])
    last_correct_num = result_all.loc[~mask, 'trial_lgbm'].max()
    # print(last_correct_num)

    wrong_num = result_all.loc[mask, 'trial_lgbm'].to_list()
    right_num = [last_correct_num + x + 1 for x in range(len(wrong_num))]
    # result_all.loc[mask, 'trial_lgbm'] = result_all.loc[mask, 'trial_lgbm'].replace(wrong_num, right_num)
    # # print(len(wrong_num), wrong_num, len(right_num), right_num)
    #
    # # 2. rename
    # result_all['name'] = result_all['name'].replace(rename['old_name'].to_list(), rename['new_name'].to_list())

    # result_all.to_csv('results_lightgbm_new.csv', index=False)

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

def new_feature_importance():

    last_correct_num, wrong_num, right_num = new_results_lightgbm()
    print(wrong_num, right_num)

    try:
        feature = pd.read_csv('results_feature_importance.csv', low_memory=False)
        print('local version run - results_feature_importance')

    except:
        print('-----------------> downloading data from DB TABLE results_feature_importance')

        with engine.connect() as conn:
            feature = pd.read_sql('SELECT * FROM results_feature_importance', conn)
        engine.dispose()

        feature.to_csv('results_feature_importance.csv', index=False)

    start = feature.loc[feature['trial_lgbm']==240687].index[0]
    end = feature.loc[feature['trial_lgbm']==9445].index[-1]+1
    print(start, end)

    feature[start:end] = feature[start:end].replace(wrong_num, right_num)

    feature.to_csv('results_feature_importance_new.csv', index=False)

def label_x_type():
    ''' add column x_type to DB TABLE results_lightgbm '''

    # read TABLE results_feature_importance
    feature = pd.read_csv('results_feature_importance_new.csv')
    print(feature.shape)

    # find null columns for each row in results_feature_importance TABLE
    col_name_arr = pd.DataFrame([feature.columns.to_list()] * len(feature))
    null_col_name_arr = col_name_arr.mask(~feature.isnull().values, np.nan).replace(np.nan, '').values
    null_col_list = [''.join(x) for x in null_col_name_arr]

    null_col = {}       # different groups of columns that may have null
    null_col['ni'] = ['ni_ts01', 'ni_ts13', 'ni_ts35']      # when using ibes eps_ts01 ... instead
    null_col['fwd'] = ['fwd_ey', 'fwd_roic']                # fwd ratios with ibes + ws data
    null_col['eps'] = ['eps_ts01', 'eps_ts13', 'eps_ts35']  # yoy for ibes eps ttm data
    null_col['qcut'] = ['ibes_qcut_as_x']                   # add consensus qcut as x

    for k in null_col.keys():           # create list of null columns by
        v = (''.join(null_col[k]))
        null_col_list = [x.replace(v, k) for x in null_col_list]

    feature['x_type'] = null_col_list

    info = pd.read_csv('result_all_new.csv', low_memory=False)
    print('old results_lightgbm shape: ', info.shape)

    feature_info = info.merge(feature[['trial_lgbm', 'x_type']], on='trial_lgbm', how='left').drop_duplicates()
    feature_info = feature_info.drop_duplicates(keep='first') # for unsolved duplicates where name='ni_new inudstry_qcut x'

    feature_info = feature_info.sort_values('trial_lgbm')
    # feature_info['x_type'] = feature_info.groupby(['name','trial_hpot']).fillna(method='ffill')['x_type']
    # feature_info['x_type'] = feature_info.groupby(['name','trial_hpot']).fillna(method='bfill')['x_type']

    feature_info['y_type'] = feature_info['y_type'].fillna('ni')
    feature_info['exclude_fwd'] = feature_info['exclude_fwd'].fillna(False)

    # feature_info[['name', 'x_type', 'y_type']].drop_duplicates(subset=['name', 'x_type', 'y_type']).to_csv('###.csv', index=False)

    feature_info.loc[(feature_info['name']=='ibes_new industry') & (feature_info['y_type']=='ni'), 'name'] = 'ni_new industry_qcut x'

    feature_info.to_csv('result_lightgbm_new_new.csv', index=False)
    print('new results_lightgbm shape: ', feature_info.shape)

if __name__ == '__main__':

    label_x_type()


