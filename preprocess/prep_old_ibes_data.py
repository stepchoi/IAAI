from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

from miscel import check_dup, date_type

def summary_to_data():

    cs = pd.read_csv('ibes_summary.csv')
    cs = cs.loc[(cs['MEASURE']=='EPS')&(cs['FISCALP']=='ANN')]
    cs['CUSIP'] = [str(x).zfill(8) for x in cs['CUSIP']]
    cs = date_type(cs, date_col='FPEDATS', format='%Y%m%d')
    cs = date_type(cs, date_col='STATPERS', format='%Y%m%d')

    # cs['cut_date'] = cs['FPEDATS'].apply(lambda x: x + relativedelta(days=1) - relativedelta(months=9) - relativedelta(days=1))
    # print(cs['cut_date'])
    # cs = cs.loc[cs['STATPERS'] < cs['cut_date']].groupby(['CUSIP','FPEDATS']).last().reset_index()
    print(cs)

    cs_new = pd.read_csv('preprocess/ibes_data.csv', usecols = ['identifier','period_end','EPS1FD12','EPS1TR12'])
    cs_new['identifier_1'] = [str(x)[:-1] for x in cs_new['identifier']]
    cs_new = date_type(cs_new)

    compare = cs.merge(cs_new, left_on=['CUSIP','FPEDATS'], right_on=['identifier_1','period_end'], how='inner')
    # compare = compare.filter(['identifier','FPEDATS','MEDEST','ACTUAL'])
    # compare.columns = ['identifier','period_end','EPS1FD12','EPS1TR12']

    compare.to_csv('ibes_data_old_all.csv', index=False)


if __name__ == '__main__':
    summary_to_data()
    # data_to_yoy()