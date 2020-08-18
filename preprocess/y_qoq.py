from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt

from miscel import check_dup, date_type

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def clean_ibes_excel():



if __name__ == '__main__':
