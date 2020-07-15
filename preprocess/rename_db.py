import numpy as np
import pandas as pd
from sqlalchemy import create_engine


db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

if __name__ == '__main__':

    try:
        result_all = pd.read_csv('result_all.csv')

    except:
        with engine.connect() as conn:
            result_all = pd.read_sql('SELECT * FROM results_lightgbm', conn)
        engine.dispose()

        result_all.to_csv('result_all.csv', index=False)

    rename = pd.read_excel('preprocess/ratio_calculation.xlsx','DB_name')
    print(rename)


