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

def date_type(df, date_col='period_end', format=''):
    ''' convert period_end columns to datetime type '''

    if format == '':
        fmt_list = ['%d/%m/%Y', '%Y-%m-%d', '%Y%m%d']
    else:
        fmt_list = [format]

    for fmt in fmt_list:
        try:
            df[date_col] = pd.to_datetime(df[date_col], format=fmt)
            break
        except:
            continue

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

def reorder_col(df, first_cols):
    col = set(df.columns.to_list())
    return df[first_cols + list(col-set(first_cols))]

if __name__ == '__main__':

    # init_nodes = 16
    # nodes_mult = 1
    # mult_freq = 1
    # mult_start = 2
    #
    # for i in range(6):
    #
    #     temp_nodes = int(min(init_nodes * (2 ** (nodes_mult * max((i - mult_start+3)//mult_freq, 0))), 128)) # nodes grow at 2X or stay same - at most 128 nodes
    #     # print((i-mult_start+1)//mult_freq)
    #     print(i, temp_nodes)

    import datetime as dt
    from dateutil.relativedelta import relativedelta

    ll = pd.read_csv('ll.csv', index_col='Unnamed: 0')
    ll = ll.fillna(0).to_dict()
    lll = {}

    for i in ll.keys():
        lll[i] = {}
        for t in ll[i].keys():
            if ll[i][t] < 5:
                lll[i][t] = 1

    print(lll)





