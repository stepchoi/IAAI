from sqlalchemy import create_engine
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import datetime as dt

def download(update=0):
    ''' donwload results from results_lightgbm '''

    if update==0:   # update if newer results is downloaded
        results = pd.read_csv('results_lgbm/params_tuning/results_{}.csv'.format(dt.datetime.now().strftime('%Y%m%d')))
        print('local version run - results')
    elif update==1:
        with engine.connect() as conn:
            results = pd.read_sql('SELECT * FROM results_lightgbm WHERE max_bin IS NOT NULL', con=conn)
        engine.dispose()

        results.to_csv('results_lgbm/params_tuning/results_{}.csv'.format(dt.datetime.now().strftime('%Y%m%d')), index=False)

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

def plot_boxplot(df, only_test=False):
    ''' plot distribution of mae based on different hyper-parameters'''

    params = 'bagging_fraction, bagging_freq, feature_fraction, lambda_l1, learning_rate, min_data_in_leaf, ' \
             'min_gain_to_split, lambda_l2, boosting_type, max_bin, num_leaves, name, exclude_fwd'.rsplit(', ')

    df = df.loc[df['name']!='classification']

    n = round(np.sqrt(len(params)))+1
    fig = plt.figure(figsize=(4*n, 4*n), dpi=120)
    # fig.suptitle(subset_name, fontsize=14)

    k = 1
    des_df_list = []
    print(df.describe().T)
    for p in params:
        print(p, set(df[p]))
        #
        # des_df = pd.DataFrame()
        # print(df.groupby(p).mean()[['mae_train','mae_valid','mae_test']])

        data_test = []
        data_train = []
        label = []
        for name, g in df.groupby([p]):
            print(g)

            label.append(name)
            data_test.append(g['mae_test'])
            data_train.append(g['mae_train'])

        ax = fig.add_subplot(n, n, k)

        def draw_plot(ax, data, label, edge_color, fill_color):
            bp = ax.boxplot(data, labels = label, patch_artist=True)

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=edge_color)

            for patch in bp['boxes']:
                patch.set(facecolor=fill_color)

        draw_plot(ax, data_test, label, 'red', 'tan')

        if only_test == False:
            draw_plot(ax, data_train, label, 'blue', 'cyan')

        ax.set_title(p)
        k += 1
    name = {True: 'test', False: 'all'}
    fig.savefig('results_lgbm/params_tuning/plot_{}_{}.png'.format(dt.datetime.now().strftime('%Y%m%d'), name[only_test]))

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

    results = download(0)
    # calc_correl(results)
    plot_boxplot(results, only_test=True)
    plot_boxplot(results, only_test=False)

    # compare_valid()