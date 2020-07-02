from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def download():
    ''' donwload results from results_lightgbm '''

    with engine.connect() as conn:
        results = pd.read_sql('SELECT * FROM results_lightgbm WHERE max_bin IS NOT NULL', con=conn)
    engine.dispose()

    print(', '.join([results.columns.to_list()]))
    print(results)

    results.to_csv('results_2Jun.csv', index=False)


if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    download()