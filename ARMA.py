import pmdarima as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import r2_score, mean_absolute_error

from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from load_data_arima import load_data

db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
engine = create_engine(db_string)

def auto_arma(arr):
    model = pm.auto_arima(arr[:-1], start_p=1, start_q=1,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # maximum p and q
                          m=1,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=False,  # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    # print(model.summary())

    # model.plot_diagnostics(figsize=(7, 5))
    # plt.show()

    # Forecast
    n_periods = 1
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = np.arange(len(arr), len(arr) + n_periods)

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    # lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    # upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # # Plot
    # plt.plot(arr)
    # plt.plot(fc_series, color='darkgreen')
    # plt.fill_between(lower_series.index,
    #                  lower_series,
    #                  upper_series,
    #                  color='k', alpha=.15)
    #
    # plt.show()

    return abs(arr[-1] - fc_series.values[0])

def auto_arma_all(train_x):

    mae = []
    for i in range(len(train_x)):
        try:
            mae.append(auto_arma(train_x.values[i]))
        except:
            mae.append(np.nan)

        print(i, np.mean(mae))

    print(mae)
    pd.DataFrame(mae, index=0).to_csv('mae.csv', index=False)

if __name__ == "__main__":

    sql_result = {}

    # default params for load_data
    period_1 = dt.datetime(2013,3,31)
    sample_no = 25

    sql_result['eps_only'] = True

    # these are parameters used to load_data
    sql_result['name'] = 'arma'
    data = load_data()

    data.split_entire(add_ind_code=0)
    sql_result['icb_code'] = 0

    for i in tqdm(range(sample_no)):  # roll over testing period
        testing_period = period_1 + i * relativedelta(months=3)
        sql_result['testing_period'] = testing_period

        train_x = data.split_train_test(testing_period)
        train_x = train_x.loc[~train_x.iloc[:,-1].isnull()]
        train_x = train_x.fillna(0)

        auto_arma_all(train_x)
        break

