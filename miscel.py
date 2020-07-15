import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

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

def write_db(df, table_name):
    ''' write to db by chucksize 10000 and and pregress bar to keep track '''

    def chunker(seq, size=1000):
        return (seq[pos:pos + size] for pos in np.arange(0, len(seq), size))

    with engine.connect() as conn:
        with tqdm(total=len(df)) as pbar:
            for i, cdf in enumerate(chunker(df, 1000)):
                replace = "replace" if i == 0 else "append"
                df.to_sql(table_name, con=conn, index=False, if_exists=replace, chunksize=1000, method='multi')
                pbar.update(1000)
    engine.dispose()

if __name__ == '__main__':
    x = pd.read_csv('##load_data_qcut_train.csv', usecols=['period_end','identifier','ibes_qcut_as_x'])
    # x = pd.read_csv('#check_sector.csv', usecols=['period_end','identifier','ibes_qcut_as_x'])
    print(date_type(x).dtypes)
    y = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv')
    print(date_type(y).dtypes)
    m = pd.merge(date_type(x), date_type(y), on=['period_end','identifier'], how='left')
    print(m.isnull().sum())
    m.to_csv('#check_cut1.csv', index=False)

