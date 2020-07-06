from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from results_lgbm.consensus import eps_to_yoy

def qcut_yoy(yoy):
    ''' convert yoy in qcut format to medians with med_train from training set'''

    with engine.connect() as conn:
        bins_df = pd.read_sql("SELECT * FROM results_bins WHERE med_train ='{\"Not applicable\"}'", conn)
    engine.dispose()

    yoy_list = []
    for i in tqdm(range(len(bins_df))):   # roll over all cut_bins used by LightGBM -> convert to median

        if bins_df.iloc[i]['icb_code'] == 999999:   # represent miscellaneous model
            indi_models = [301010, 101020, 201030, 302020, 351020, 502060, 552010, 651010, 601010, 502050, 101010,
                           501010, 201020, 502030, 401010]
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (~yoy['icb_sector'].isin(indi_models))]

        else:
            part_yoy = yoy.loc[(yoy['period_end'] == bins_df.iloc[i]['testing_period']) &
                               (yoy['icb_sector'] == bins_df.iloc[i]['icb_code'])]

        cut_bins = bins_df.iloc[i]['cut_bins'].strip('{}').split(',')  # convert string {1, 2, 3....} to list
        cut_bins[0], cut_bins[-1] = -np.inf, np.inf  # convert cut_bins into [-inf, ... , inf]
        cut_bins = [float(x) for x in cut_bins]  # convert string in list to float

        # cut original series into 0, 1, .... (bins * n)
        part_yoy['y_ibes_qcut'] = pd.cut(part_yoy['y_ibes'], bins=cut_bins, labels=False)
        part_yoy['y_ni_qcut'] = pd.cut(part_yoy['y_ni'], bins=cut_bins, labels=False)

        yoy_list.append(part_yoy)

    yoy_qcut = pd.concat(yoy_list, axis=0)

    return yoy_qcut

def act_lgbm_ibes(yoy_ibes_median, update):
    ''' combine all prediction together '''

    if update !=1 :     # No Update
        detail_stock = pd.read_csv('results_lgbm/compare_with_ibes/ibes_class_stock.csv')
        print('local version run - detail_stock')

    elif update == 1:   # Update stock specific results from FB
        print('----------------> update stock results from DB ')

        with engine.connect() as conn:  #
            result_stock = pd.read_sql('SELECT * FROM results_lightgbm_stock WHERE trial_lgbm > 81591', conn)
            trial_lgbm = set(result_stock['trial_lgbm'])

            query = text("SELECT trial_lgbm, qcut_q, icb_code, testing_period, cv_number, mae_test, exclude_fwd "
                         "FROM results_lightgbm WHERE name='classification' AND (trial_lgbm IN :trial_lgbm)")
            query = query.bindparams(trial_lgbm=tuple(trial_lgbm))
            result_all = pd.read_sql(query, conn)

        engine.dispose()

        detail_stock = result_stock.merge(result_all, on=['trial_lgbm'], how='inner')
        detail_stock.to_csv('results_lgbm/compare_with_ibes/ibes_class_stock.csv', index=False)

    # convert datetime
    detail_stock['testing_period'] = pd.to_datetime(detail_stock['testing_period'], format='%Y-%m-%d')
    yoy_ibes_median['period_end'] = pd.to_datetime(yoy_ibes_median['period_end'], format='%Y-%m-%d')

    detail_stock = detail_stock.filter(['identifier', 'qcut_q', 'icb_code', 'testing_period', 'cv_number', 'pred'])

    # merge (stock prediction) with (ibes consensus median)
    yoy_merge = detail_stock.merge(yoy_ibes_median, left_on=['identifier', 'testing_period'],
                                      right_on=['identifier', 'period_end'])

    return yoy_merge

def calc_accu(yoy_merge):
    ''' calculate mae for each testing_period, icb_code, (cv_number) '''

    def part_accu(df):
        ''' calculate different mae for groups of sample '''

        dict = {}
        dict['ibes'] = accuracy_score(df['y_ibes_qcut'], df['y_ni_qcut'])
        dict['lgbm'] = accuracy_score(df['pred'], df['y_ni_qcut'])
        dict['len'] = len(df)
        return dict

    mae = {}
    for p in set(yoy_merge['testing_period']):
        part_p = yoy_merge.loc[yoy_merge['testing_period']==p]

        for i in set(yoy_merge['icb_code']):
            part_i = part_p.loc[part_p['icb_code']==i]

            try:    # calculate aggregate mae for all 5 cv groups
                mae['{}_{}'.format(p, i)] = part_accu(part_i)
            except:
                print('not available', p, i)
                continue

    df = pd.DataFrame(mae).T.reset_index()

    df[['testing_period', 'icb_code']] = df['index'].str.split('_', expand=True)
    df = df.filter(['icb_code','testing_period', 'cv_number','ibes','lgbm','len'])

    df.to_csv('results_lgbm/compare_with_ibes/ibes_class_accuracy.csv', index=False)

def main(update=0):
    ''' main function: clean ibes + calculate mae '''

    if update == 1:

        try:
            yoy = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv')
            yoy['period_end'] = pd.to_datetime(yoy['period_end'], format='%Y-%m-%d')
            print('local version run - ibes_yoy ')
        except:
            yoy = eps_to_yoy().merge_and_calc()
            yoy.to_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv', index=False)

        yoy_med = qcut_yoy(yoy)                             # Update every time for new cut_bins
        yoy_merge = act_lgbm_ibes(yoy_med, update)          # merge ibes / actual / lgbm predictions
        yoy_merge.to_csv('results_lgbm/compare_with_ibes/ibes_yoy_merge_class.csv', index=False)

    else:
        yoy_merge = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy_merge_class.csv')

    calc_accu(yoy_merge)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    main(1)


    # with engine.connect() as conn:
    #     df = pd.read_sql('SELECT * FROM results_lightgbm WHERE trial_lgbm >= 81591', conn)
    # engine.dispose()
    #
    # print(df)
    #
    # print(df.describe().T)



