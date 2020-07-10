from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt

from miscel import check_dup, date_type
from preprocess.ratios import worldscope

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

'''
1. read DB TABLE
2. fill 0
3. pivot 3D array
'''

def read_data():

    ''' read worldscope_quarter_summary / ibes_data / stock_data '''

    try:  # read Worldscope Data after cleansing
        ws = pd.read_csv('preprocess/quarter_summary_clean.csv')
        ibes = pd.read_csv('preprocess/ibes_data.csv')
        stock = pd.read_csv('preprocess/stock_data.csv')
        print('local version run - quarter_summary_clean / ibes_data / stock_data ')
    except:
        ws = worldscope().fill_missing_ws()