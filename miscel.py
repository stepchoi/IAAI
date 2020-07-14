import numpy as np
import pandas as pd
from sqlalchemy import create_engine


db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

# Debug def
def check_dup(df, index_col=['identifier','period_end'], ex=True):
    dup = df.duplicated(subset=index_col, keep=False)
    df_dup = df.loc[dup].sort_values(index_col)
    df_dup.to_csv('#check_dup.csv', index=False)
    print(df_dup)

    if ex == True:
        print('exit from check_dup')
        exit(0)

def date_type(df, date_col='period_end'):
    ''' convert period_end columns to datetime type '''

    try:
        df[date_col] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
    except:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')

    return df



if __name__ == '__main__':
    pass