from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import datetime as dt

def download():
    ''' donwload results from results_lightgbm '''

    try:
        results = pd.read_csv('results_lgbm/results_20200702.csv')   # update if newer results is downloaded
        print('local version run - results')
    except:
        with engine.connect() as conn:
            results = pd.read_sql('SELECT * FROM results_lightgbm WHERE max_bin IS NOT NULL', con=conn)
        engine.dispose()

        results.to_csv('results_lgbm/results_{}.csv'.format(dt.datetime.now().strftime('%Y%m%d')), index=False) # download results & label with date

    # print(', '.join(results.columns.to_list()))
    print(results.columns)

    return results

def calc_correl(results):
    correls = {}
    correls['train_valid'] = {}
    correls['valid_test'] = {}

    correls['train_valid']['all'] = results['mae_train'].corr(results['mae_valid'])
    correls['valid_test']['all'] = results['mae_valid'].corr(results['mae_test'])

    for t in set(results['testing_period']):
        str_t = pd.Timestamp(t).strftime('%Y%m%d')
        for i in set(results['icb_code']):
            part = results.loc[(results['testing_period'] == t) & (results['icb_code'] ==i)]
            correls['train_valid']['{}_{}'.format(str_t, i)] = part['mae_train'].corr(part['mae_valid'])
            correls['valid_test']['{}_{}'.format(str_t, i)] = part['mae_valid'].corr(part['mae_test'])


    print(pd.DataFrame(correls))

    pd.DataFrame(correls).to_csv('results_lgbm/results_correl.csv')




if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    results = download()
    calc_correl(results)