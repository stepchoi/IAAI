from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt

def download(r_name, best='best'):
    ''' donwload results from results_lightgbm '''

    if best == 'best':
        query = text("select * from (select DISTINCT *, min(mae_valid) over (partition by trial_hpot) as min_thing " \
                "from results_{} where name in :name) t where mae_valid = min_thing".format(tname))
        query = query.bindparams(name=tuple(r_name))
    elif r_name == 'all':
        query = 'SELECT * FROM results_{}'.format(tname)
    else:
        query = text("SELECT * FROM results_{} WHERE name in :name".format(tname))
        query = query.bindparams(name=tuple(r_name))

    print('--------> download from DB TABLE')
    with engine.connect() as conn:
        results = pd.read_sql(query, con=conn)
    engine.dispose()
    print(results.columns)

    results.to_csv('results_analysis/params_tuning/lgbm_{}|{}.csv'.format(best, r_name), index=False)

    calc_correl(results) # check correlation

    return results

def calc_correl(results):
    ''' calculated correlation between train/valid/test sets '''

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

    # pd.DataFrame(correls).to_csv('results_analysis/params_tuning/results_correl.csv')

def calc_average(df, r_name, model='lgbm'):
    ''' calculate mean of each variable in db '''

    if tname == 'lightgbm':
        params = ['bagging_fraction', 'bagging_freq', 'feature_fraction', 'lambda_l1', 'learning_rate',
                  'min_data_in_leaf', 'min_gain_to_split', 'lambda_l2', 'boosting_type', 'max_bin', 'num_leaves']
    elif tname == 'xgboost':
        params = ['alpha', 'booster', 'colsample_bylevel', 'colsample_bycode', 'colsample_bytree', 'eta', 'gamma', 'lambda',
                  'max_bin', 'max_depth', 'min_child_weight', 'num_leaves', 'subsample']
    elif tname == 'randomforest':
        params = ['max_features', 'max_samples', 'min_impurity_decrease', 'min_samples_leaf',
                  'min_samples_split', 'min_weight_fraction_leaf','max_depth']

    writer = pd.ExcelWriter('results_analysis/params_tuning/{}_describe|{}.xlsx'.format(model, r_name))    # create excel records

    for c in set(df['icb_code']):
        sub_df = df.loc[df['icb_code']==c]

        df_list = []
        # df_list.append(sub_df.mean()['mae_test'])
        for p in params:
            try:
                des_df = sub_df.groupby(p).mean()['mae_test'].reset_index()  # calculate means of each subset
                des_df['len'] = sub_df.groupby(p).count().reset_index()['mae_test']
                des_df.columns = ['subset', 'mae_test', 'len']
                des_df['params'] = p
                des_df = des_df.sort_values(by=['mae_test'], ascending=True)
                print(des_df)

                # std_arr = df.groupby(p).std()[['mae_train', 'mae_valid', 'mae_test']].values  # calculate std of each subset
                # des_df[['mae_train_std', 'mae_valid_std', 'mae_test_std']] = pd.DataFrame(std_arr)
                df_list.append((des_df))
            except:
                continue
        pd.concat(df_list, axis=0).to_excel(writer, '{}'.format(c), index=False)

    writer.save()

def plot_boxplot(df, r_name=None):
    ''' plot distribution of mae based on different hyper-parameters'''

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
    ''' compare chronological order and validation order '''

    chron_v = pd.read_csv('results_analysis/results_chron_valid.csv', usecols=['icb_code', 'testing_period', 'mae_train', 'trial_hpot',
                                                                         'mae_valid', 'mae_test', 'exclude_fwd'])
    group_v = pd.read_csv('results_analysis/results_regression.csv', usecols=['icb_code', 'testing_period', 'mae_train', 'trial_hpot',
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
    final.to_csv('results_analysis/compare_valid.csv', index=False)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    # r_name = ['ibes_entire_only ws -smaller space','ibes_entire_only ws -smaller space','ibes_entire_only ws -smaller space']
    r_name = ['xgb tuning -sample_type industry -x_type fwdepsqcut']
    r_name = ['xgb tuning3 -sample_type entire -x_type fwdepsqcut']
    # r_name = ['ibes_new industry_only ws -indi space3']
    r_name = ['rf extratree -sample_type entire -x_type fwdepsqcut']
    r_name = ['xgb ind -sample_type industry -x_type fwdepsqcut']
    r_name = ['ibes_new industry_only ws -indi space3']
    r_name = ['ibes_entire_only ws -smaller space']
    r_name = ['xgb ind_all_tuning -sample_type industry -x_type ni']

    if 'xgb' in r_name[0]:
        tname = 'xgboost'
    elif 'rf' in r_name[0]:
        tname = 'randomforest'
    else:
        tname = 'lightgbm'

    results = download(r_name=r_name, best='best')
    calc_average(results, r_name)
    # plot_boxplot(results, r_name=r_name)

    # calc_correl(results)
    # compare_valid()