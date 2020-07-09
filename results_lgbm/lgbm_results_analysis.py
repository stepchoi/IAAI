from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

def download(update=0, r_name = None):
    ''' donwload results from results_lightgbm '''

    if update==0:   # update if newer results is downloaded
        results = pd.read_csv('results_lgbm/params_tuning/results_{}.csv'.format(r_name))
        print('local version run - results -', r_name)

    elif update==1:
        with engine.connect() as conn:
            if r_name != None:
                results = pd.read_sql("SELECT * FROM results_lightgbm WHERE name = '{}'".format(r_name), con=conn)
            else:
                results = pd.read_sql('SELECT * FROM results_lightgbm WHERE max_bin IS NOT NULL', con=conn)
        engine.dispose()

        results.to_csv('results_lgbm/params_tuning/results_{}.csv'.format(r_name), index=False)

    # print(', '.join(results.columns.to_list()))
    # print(results.columns)
    return results

def calc_correl(results):
    correls = {}
    correls['train_valid'] = {}
    correls['valid_test'] = {}

    correls['train_valid']['all'] = results['mae_train'].corr(results['mae_valid'])
    correls['valid_test']['all'] = results['mae_valid'].corr(results['mae_test'])

    for t in set(results['testing_period']):
        str_t = pd.Timestamp(t).strftime('%Y%m%d')
        for i in set(results['icb_code']):
            part = results.loc[(results['testing_period'] == t) & (results['icb_code'] ==i)]
            correls['train_valid']['{}_{}'.format(str_t, i)] = part['mae_train'].corr(part['mae_valid'])
            correls['valid_test']['{}_{}'.format(str_t, i)] = part['mae_valid'].corr(part['mae_test'])


    print(pd.DataFrame(correls))

    pd.DataFrame(correls).to_csv('results_lgbm/params_tuning/results_correl.csv')

def plot_boxplot(df, r_name=None):
    ''' plot distribution of mae based on different hyper-parameters'''

    params = 'bagging_fraction, bagging_freq, feature_fraction, lambda_l1, learning_rate, min_data_in_leaf, ' \
             'min_gain_to_split, lambda_l2, boosting_type, max_bin, num_leaves, name, exclude_fwd'.rsplit(', ')

    n = round(np.sqrt(len(params)))+1

    fig_test = plt.figure(figsize=(4*n, 4*n), dpi=120)      # create figure for test only boxplot
    fig_all = plt.figure(figsize=(4*n, 4*n), dpi=120)       # create figure for test & train boxplot
    fig_test.suptitle('{}_test'.format(r_name), fontsize=14)
    fig_all.suptitle('{}_all'.format(r_name), fontsize=14)

    k = 1
    des_df_list = []
    print(df.describe().T)
    for p in params:
        print(p, set(df[p]))

        # calculate mean of each variable in db
        des_df = df.groupby(p).mean()[['mae_train','mae_valid','mae_test']].reset_index()   # calculate means of each subset
        des_df.columns = ['subset', 'mae_train_avg', 'mae_valid_avg', 'mae_test_avg']

        std_arr = df.groupby(p).std()[['mae_train','mae_valid','mae_test']].values           # calculate std of each subset
        des_df[['mae_train_std', 'mae_valid_std', 'mae_test_std']] = pd.DataFrame(std_arr)

        des_df = des_df.sort_values(by=['mae_test_avg'], ascending=True)
        des_df['params'] = p
        des_df_list.append(des_df.filter(['params', 'subset', 'mae_train_avg', 'mae_valid_avg', 'mae_test_avg',
                                          'mae_train_std', 'mae_valid_std', 'mae_test_std']))

        # prepare for plotting
        data_test = []
        data_train = []
        label = []
        for name, g in df.groupby([p]):
            label.append(name)
            data_test.append(g['mae_test'])
            data_train.append(g['mae_train'])

        ax_test = fig_test.add_subplot(n, n, k)
        ax_all = fig_all.add_subplot(n, n, k)

        def draw_plot(ax, data, label, edge_color, fill_color):
            bp = ax.boxplot(data, labels = label, patch_artist=True)

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)

            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        draw_plot(ax_test, data_test, label, 'red', 'tan')
        draw_plot(ax_all, data_test, label, 'red', 'tan')
        draw_plot(ax_all, data_train, label, 'blue', 'cyan')

        ax_test.set_title(p)
        ax_all.set_title(p)
        k += 1

    fig_test.savefig('results_lgbm/params_tuning/plot_{}_test.png'.format(r_name))
    fig_all.savefig('results_lgbm/params_tuning/plot_{}_all.png'.format(r_name))
    pd.concat(des_df_list, axis=0).to_csv('results_lgbm/params_tuning/results_{}_describe.csv'.format(r_name), index=False)

def compare_valid():
    chron_v = pd.read_csv('results_lgbm/results_chron_valid.csv', usecols=['icb_code', 'testing_period', 'mae_train', 'trial_hpot',
                                                                         'mae_valid', 'mae_test', 'exclude_fwd'])
    group_v = pd.read_csv('results_lgbm/results_regression.csv', usecols=['icb_code', 'testing_period', 'mae_train', 'trial_hpot',
                                                                        'mae_valid', 'mae_test', 'exclude_fwd'])
    def clean(df, valid_type):
        df_best = df.iloc[df.groupby(['trial_hpot'])['mae_valid'].idxmin()]
        df_mean = df_best.groupby(['icb_code', 'testing_period','exclude_fwd']).mean()[['mae_train', 'mae_valid', 'mae_test']]
        df_mean = df_mean.reset_index(drop=False)
        df_mean['valid_type'] = valid_type
        return df_mean

    df_list = []
    df_list.append(clean(chron_v, 'chron'))
    df_list.append(clean(group_v, 'stock'))
    final = pd.concat(df_list, axis=0)
    final.to_csv('results_lgbm/compare_valid.csv', index=False)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    r_name = 'industry'

    results = download(0, r_name=r_name)
    plot_boxplot(results, r_name=r_name)

    # calc_correl(results)
    # compare_valid()