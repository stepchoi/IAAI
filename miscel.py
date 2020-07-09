import numpy as np
import pandas as pd

# Debug def
def check_dup(df, index_col=['identifier','period_end'], ex=True):
    dup = df.duplicated(subset=index_col, keep=False)
    print(df.loc[dup].sort_values(index_col))

    if ex == True:
        print('exit from check_dup')
        df.loc[dup].sort_values(index_col).to_csv('#check_dup.csv', index=False)
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