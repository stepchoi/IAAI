from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
from results_analysis.lgbm_params_tuning import calc_correl, calc_average

params = ['batch_size', 'dropout', 'init_nodes', 'learning_rate', 'mult_freq', 'mult_start', 'nodes_mult',
          'num_Dense_layer', 'num_nodes']


def download(r_name, best='best'):
    ''' donwload results from results_lightgbm '''

    if best == 'best':
        query = "select * from (select DISTINCT *, min(mae_valid) over (partition by trial_hpot, exclude_fwd, icb_code) " \
                "as min_thing from results_{})t where mae_valid = min_thing".format(tname)

    try:  # update if newer results is downloaded
        print('--------> params tuning: lgbm_{}|{}.csv'.format(best, r_name))
        results = pd.read_csv('results_analysis/params_tuning/dense2_{}|{}.csv'.format(best, r_name))
        print('local version run - {}.csv'.format(r_name))

    except:
        print('--------> download from DB TABLE')
        with engine.connect() as conn:
            results = pd.read_sql(query, con=conn)
        engine.dispose()

        results.to_csv('results_analysis/params_tuning/dense2_{}|{}.csv'.format(best, r_name), index=False)

    calc_correl(results)  # check correlation

    print(results.columns)

    return results


def plot_boxplot(df, table_name='results_dense', r_name=None):
    ''' plot distribution of mae based on different hyper-parameters'''

    if r_name == None:
        r_name = table_name

    print(df.columns)

    params = ['batch_size', 'dropout_1',
              'dropout_2', 'dropout_3', 'dropout_4', 'dropout_5', 'learning_rate',
              'neurons_layer_1', 'neurons_layer_2', 'neurons_layer_3',
              'neurons_layer_4', 'neurons_layer_5', 'num_Dense_layer']

    n = round(np.sqrt(len(params))) + 1

    print(df)

    if table_name == 'results_dense':
        for i in range(1, 6):
            df.loc[(df['num_Dense_layer'] < i), ['neurons_layer_{}'.format(i), 'dropout_{}'.format(i)]] = np.nan

    print(df)

    df.to_csv('results_dense_nan.csv', index=False)
    exit(0)

    fig_test = plt.figure(figsize=(4 * n, 4 * n), dpi=120)  # create figure for test only boxplot
    fig_all = plt.figure(figsize=(4 * n, 4 * n), dpi=120)  # create figure for test & train boxplot
    fig_test.suptitle('{}_test'.format(r_name), fontsize=14)
    fig_all.suptitle('{}_all'.format(r_name), fontsize=14)

    k = 1
    des_df_list = []
    print(df.describe().T)
    for p in params:
        print(p, set(df[p]))

        # calculate mean of each variable in db
        des_df = df.groupby(p).mean()[
            ['mae_train', 'mae_valid', 'mae_test']].reset_index()  # calculate means of each subset
        des_df.columns = ['subset', 'mae_train_avg', 'mae_valid_avg', 'mae_test_avg']

        std_arr = df.groupby(p).std()[['mae_train', 'mae_valid', 'mae_test']].values  # calculate std of each subset
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
            bp = ax.boxplot(data, labels=label, patch_artist=True)

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)

            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

            ax.set(ylim=(0.005, 0.015))

        draw_plot(ax_test, data_test, label, 'red', 'tan')
        draw_plot(ax_all, data_test, label, 'red', 'tan')
        draw_plot(ax_all, data_train, label, 'blue', 'cyan')

        ax_test.set_title(p)
        ax_all.set_title(p)
        k += 1

    # fig_test.tight_layout()
    fig_all.tight_layout()

    # fig_test.savefig('results_analysis/params_tuning/plot_{}_test.png'.format(r_name))
    fig_all.savefig('results_analysis/params_tuning/plot_{}_all.png'.format(r_name))
    pd.concat(des_df_list, axis=0).to_csv('results_analysis/params_tuning/{}_describe.csv'.format(r_name), index=False)


def compare_valid():
    chron_v = pd.read_csv('results_analysis/results_chron_valid.csv',
                          usecols=['icb_code', 'testing_period', 'mae_train', 'trial_hpot',
                                   'mae_valid', 'mae_test', 'exclude_fwd'])
    group_v = pd.read_csv('results_analysis/results_regression.csv',
                          usecols=['icb_code', 'testing_period', 'mae_train', 'trial_hpot',
                                   'mae_valid', 'mae_test', 'exclude_fwd'])

    def clean(df, valid_type):
        df_best = df.iloc[df.groupby(['trial_hpot'])['mae_valid'].idxmin()]
        df_mean = df_best.groupby(['icb_code', 'testing_period', 'exclude_fwd']).mean()[
            ['mae_train', 'mae_valid', 'mae_test']]
        df_mean = df_mean.reset_index(drop=False)
        df_mean['valid_type'] = valid_type
        return df_mean

    df_list = []
    df_list.append(clean(chron_v, 'chron'))
    df_list.append(clean(group_v, 'stock'))
    final = pd.concat(df_list, axis=0)
    final.to_csv('results_analysis/compare_valid.csv', index=False)


if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    r_name = 'new with indi code -fix space'
    tname = 'rnn_eps'

    r_name = 'small_training_False_0'
    tname = 'cnn_rnn'

    results = download(r_name=r_name)
    calc_average(results, params=params, r_name=r_name, model='dense2')
    # plot_boxplot(results, r_name=r_name)

    # calc_correl(results)
    # compare_valid()




















