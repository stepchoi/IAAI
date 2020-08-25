import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy import create_engine
from datetime import datetime
import sys
from preprocess.x_ratios import full_period


db_url_prod = "postgres://postgres:askLORA20$@dlp-prod.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres"
db_url_droid = "postgres://postgres:askLORA20$@droid-test.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres"
db_url_hkpolyu = "postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres"
test_score_results_table = "test_score_results"
dl_value_universe_table = "dl_value_universe"


def get_worldscope_identifier():
    print('=== Get Identifier ===' + str(datetime.utcnow().time()))
    engine = create_engine(db_url_hkpolyu, max_overflow=-1, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        metadata = db.MetaData()
        query = f'select member_ric, identifier  from {dl_value_universe_table}'
        data = pd.read_sql(query, con=conn)
    engine.dispose()
    data = pd.DataFrame(data)
    print("Total Data = " + str(len(data)))
    return data

if __name__ == '__main__':
    ticker = pd.Series([])
    result = pd.DataFrame({'index':[], 'trading_day':[], 'ticker':[],'close':[]}, index=[])
    data = pd.read_excel("Preprocess/Stock_Price_data_raw.xlsx")
    data = data.set_index(['Close Price']).unstack().reset_index(drop=False)
    data.columns = ['ticker','period_end','close']
    data['period_end'] = data['period_end'].astype('datetime64[ns]')
    # print(data, data.drop_duplicates(subset=['ticker', 'period_end']))

    result = full_period(data, 'ticker')

    result['stock_return_1Qa'] = (result['close'].shift(-1) / result['close']) - 1
    result.loc[result.groupby('ticker').tail(1).index, 'stock_return_1Qa'] = np.nan # stock T-3 -> T0

    result['stock_return_3Qb'] = (result['close'] / result['close'].shift(3)) - 1
    result.loc[result.groupby('ticker').head(3).index, 'stock_return_3Qb'] = np.nan # stock T0 -> T1

    worldscope_identifier = get_worldscope_identifier()
    result = result.merge(worldscope_identifier, how='left', left_on=["ticker"], right_on=["member_ric"])
    result.to_csv("#check_stock.csv",index=False)
    exit(0)

    # result[['identifier', 'period_end','close']].to_csv('preprocess/stock_data.csv', index=False)

    result = result[["identifier", "period_end", "stock_return_1Qa", "stock_return_3Qb"]]
    result[[ "stock_return_1Qa", "stock_return_3Qb"]] = result[[ "stock_return_1Qa", "stock_return_3Qb"]].astype(float)

    result = result.groupby(['identifier','period_end']).mean().reset_index(drop=False)
    result.to_csv("preprocess/stock_ratios.csv",index=False)
    #print(result)
    # for col in data.columns: 
    #     print(col) 
    # worldscope_identifier = get_worldscope_identifier()
