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

def plot_arma(model):
    '''  plot ARIMA model results '''

    print(model.summary())
    model.plot_diagnostics(figsize=(7, 5))
    plt.show()

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

def auto_arma(arr):
    ''' for each identifier: fit ARIMA model with 60Q data -> predict next 1Q '''

    model = pm.auto_arima(arr[:-4], start_p=1, start_q=1,                       # find best ARIMA model by stepwise search
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

    # Forecast
    n_periods = 4   # forecast next 1Q
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)  # fc = forecast; confint is for plotting
    index_of_fc = np.arange(len(arr), len(arr) + n_periods) # use 61 as index for forecast
    fc_series = pd.Series(fc, index=index_of_fc)   # make series for plotting purpose

    return fc_series.values[3]   # calculate absolute difference

def auto_arma_all(train_x):
    ''' roll over all identifier to calculate mae '''

    mae = {}
    for i in range(len(train_x)):   # roll over each identifier
        print('---------------------------------->', i)
        try:
            pred = auto_arma(train_x.values[i])
        except:
            pred = np.nan

        mae[train_x.index[i]] = {}
        mae[train_x.index[i]]['y'] = train_x.values[i,-1]
        mae[train_x.index[i]]['pred'] = pred

    df = pd.DataFrame(mae).T    # convert list of absolute_error to dataframe
    return df

if __name__ == "__main__":

    sql_result = {}

    # default params for load_data
    period_1 = dt.datetime(2013,3,31)   # go through 25Q from 2013-03-31
    sample_no = 25

    # these are parameters used to load_data
    data = load_data()

    for i in tqdm(range(sample_no)):  # roll over testing period
        testing_period = period_1 + i * relativedelta(months=3)
        sql_result['testing_period'] = testing_period

        train_x = data.split_train_test(testing_period)     # load data for 64Q (60Q for train + 4Q for test)
        print(train_x.shape)
        train_x = train_x[~train_x.iloc[:,-1].isnull()]     # remove testing data with NaN for testing 1Q
        train_x = train_x.fillna(0)     # fill NaN with 0 for all 60Q for train

        df = auto_arma_all(train_x)
        df['testing_period'] = testing_period   # label date

        with engine.connect() as conn:  # write to DB
            df.to_sql('results_arma_median', conn, if_exists='append', method='multi')
        engine.dispose()


