from sqlalchemy import create_engine
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

def download():
    ''' donwload results from results_lightgbm '''

    try:
        results = pd.read_csv('results_lgbm/results_20200702.csv')   # update if newer results is downloaded
        print('local version run - results')
    except:
        with engine.connect() as conn:
            results = pd.read_sql('SELECT * FROM results_lightgbm WHERE max_bin IS NOT NULL', con=conn)
        engine.dispose()

        results.to_csv('results_lgbm/results_{}.csv'.format(dt.datetime.now().strftime('%Y%m%d')), index=False) # download results & label with date

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

    pd.DataFrame(correls).to_csv('results_correl.csv')

def plot_scatter(df, only_test=False):
    ''' plot distribution of mae based on different hyper-parameters'''

    params = 'bagging_fraction, bagging_freq, feature_fraction, lambda_l1, learning_rate, min_data_in_leaf, ' \
             'min_gain_to_split, lambda_l2, boosting_type, max_bin, num_leaves'.rsplit(', ')

    n = round(np.sqrt(len(params)))+1
    fig = plt.figure(figsize=(4*n, 4*n), dpi=120)
    # fig.suptitle(subset_name, fontsize=14)

    k = 1
    for p in params:
        print(p, set(df[p]))

        data_test = []
        data_train = []
        label = []
        for name, g in df.groupby([p]):
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

    fig.savefig('results_lgbm/plot_{}_only'.format(dt.datetime.now().strftime('%Y%m%d')))


if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    results = download()
    # calc_correl(results)
    plot_scatter(results, only_test=True)