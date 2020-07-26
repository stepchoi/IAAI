from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from tqdm import tqdm
from results_lgbm.lgbm_consensus import eps_to_yoy

''' classification'''

if r_name == 'classification':
    bins_df = bins_df.loc[
        bins_df['med_train'] == "{\"Not applicable\"}"]  # classification will not have median conversion
else:
    bins_df = bins_df.loc[bins_df['med_train'] != "{\"Not applicable\"}"]

    ''' _______________________-----------------------------------______________________________'''

def qcut_yoy(yoy):
    ''' convert yoy in qcut format to medians with med_train from training set'''

    with engine.connect() as conn:
        bins_df = pd.read_sql("SELECT * FROM results_bins WHERE med_train ='{\"Not applicable\"}'", conn)
    engine.dispose()

    yoy_list = []
    for i in tqdm(range(len(bins_df))):   # roll over all cut_bins used by LightGBM -> convert to median
        print(bins_df.iloc[i]['cut_bins'])

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
        dict['ibes'] = accuracy_score(df['y_ibes_qcut'], df['y_ibes_act'])
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

    ''' main function: clean ibes + calculate mae '''

    try:    # STEP1: download ibes_data and organize to YoY
        yoy = pd.read_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv')
        yoy['period_end'] = pd.to_datetime(yoy['period_end'], format='%Y-%m-%d')
        print('local version run - 1. ibes_yoy ')
    except:
        yoy = eps_to_yoy().merge_and_calc()
        yoy.to_csv('results_lgbm/compare_with_ibes/ibes_yoy.csv', index=False)

    try:    # STEP2: convert ibes YoY to qcut / median
        yoy_med = pd.read_csv('results_lgbm/compare_with_ibes/ibes2_yoy_median.csv')
        print('local version run - 2. ibes_median ')
    except:
        yoy_med = yoy_to_median(yoy, industry)  # Update every time for new cut_bins
        yoy_med.to_csv('results_lgbm/compare_with_ibes/ibes2_yoy_median.csv', index=False)

    try:    # STEP3: download lightgbm results for stocks
        detail_stock = pd.read_csv('results_lgbm/compare_with_ibes/ibes3_detail_stock.csv')
        detail_stock['testing_period'] = pd.to_datetime(detail_stock['testing_period'], format='%Y-%m-%d')
        print('local version run - 3. detail_stock')
    except:
        detail_stock = download_result_stock()
        detail_stock.to_csv('results_lgbm/compare_with_ibes/ibes3_detail_stock.csv', index=False)

    try:    # STEP4: combine lightgbm and ibes results
        yoy_merge = pd.read_csv('results_lgbm/compare_with_ibes/ibes4_yoy_merge.csv')   # delete this file for stock results update
        print('local version run - 4. yoy_merge')
    except:
        yoy_merge = act_lgbm_ibes(detail_stock, yoy_med)
        yoy_merge.to_csv('results_lgbm/compare_with_ibes/ibes4_yoy_merge.csv', index=False)

    df = calc_mae(yoy_merge, industry, ibes_act)  # STEP5: calculate MAE

    name = {True:{True:'ibesttm_industry', False:'ibesttm_sector'}, False: {True:'wsttm_industry', False:'wsttm_sector'}}
    print('save to file name: ibes5_mae_{}'.format(name[ibes_act][industry]))
    df.to_csv('results_lgbm/compare_with_ibes/ibes5_mae_{}.csv'.format(name[ibes_act][industry]), index=False)

if __name__ == "__main__":
    db_string = 'postgres://postgres:DLvalue123@hkpolyu.cgqhw7rofrpo.ap-northeast-2.rds.amazonaws.com:5432/postgres'
    engine = create_engine(db_string)

    main()
    exit(0)

    # with engine.connect() as conn:
    #     df = pd.read_sql('SELECT * FROM results_lightgbm WHERE trial_lgbm >= 81591', conn)
    # engine.dispose()
    #
    # print(df)
    #
    # print(df.describe().T)
    try:
        df_org = pd.read_csv('preprocess/ibes_data.csv', usecols=['identifier','period_end','EPS1FD12', 'EPS1TR12']).dropna(how='any')
    except:
        with engine.connect() as conn:
            df_org = pd.read_sql('SELECT identifier, period_end, eps1fd12, eps1tr12 FROM ibes_data', conn)
        engine.dispose()

    df_org['ibes_cut'], cut_bins = pd.qcut(df_org['EPS1TR12'], q=3, retbins=True, labels=False)
    df_org['ibes_ttm_cut'] = pd.cut(df_org['EPS1FD12'], bins=cut_bins, labels=False)
    # print(df_org)
    # print('acc1: ',accuracy_score(df_org['Y_fwd'], df_org['Y_act']))


    mcap = pd.read_csv('preprocess/quarter_summary_clean.csv', usecols=['identifier','period_end','fn_8001','fn_18263', 'fn_5192'])   # 5085 / 2999 / 18263
    # mcap['fn_8001'] = mcap['fn_8001']*10e-5

    df = df_org.merge(mcap, on=['identifier','period_end'])
    df['Y_fwd'] = (df['EPS1FD12'] - df['EPS1TR12'])/df['fn_8001']*df['fn_5192']
    df['Y_fwd_ws'] = (df['EPS1FD12']/df['EPS1TR12']*df['fn_18263'] - df['fn_18263'])/df['fn_8001']

    from preprocess.ratios import full_period
    df = full_period(df, 'identifier','%Y-%m-%d')

    df['actual_eps'] = df['EPS1TR12'].shift(-4)
    df['actual_eps_ws'] = df['fn_18263'].shift(-4)

    df.loc[df.groupby('identifier').tail(4).index, ['actual_eps','fn_18263']] = np.nan  # y-1 ~ y0
    df['Y_act'] = (df['actual_eps'] - df['EPS1TR12'])/df['fn_8001']
    df['Y_act_ws'] = (df['actual_eps_ws'] - df['fn_18263'])/df['fn_8001']*df['fn_5192']

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='any')

    print('mae',mean_absolute_error(df['Y_fwd'],df['Y_act']))
    print('mae',mean_absolute_error(df['Y_fwd_ws'],df['Y_act_ws']))


    df['Y_act_cut'], cut_bins = pd.qcut(df['Y_act'], q=3, retbins=True, labels=False)
    cut_bins[0], cut_bins[-1] = -np.inf, np.inf
    df['Y_fwd_cut'] = pd.cut(df['Y_fwd'], bins=cut_bins, labels=False)

    df['Y_act_ws_cut'], cut_bins = pd.qcut(df['Y_act_ws'], q=3, retbins=True, labels=False)
    cut_bins[0], cut_bins[-1] = -np.inf, np.inf
    df['Y_fwd_ws_cut'] = pd.cut(df['Y_fwd_ws'], bins=cut_bins, labels=False)

    # print(cut_bins)
    # print(df)
    # df.to_csv('#check_ibes.csv')

    print('acc1',accuracy_score(df['ibes_ttm_cut'], df['ibes_cut']))
    print('acc2',accuracy_score(df['Y_act_cut'], df['Y_fwd_cut']))
    print('acc3',accuracy_score(df['Y_act_ws_cut'], df['Y_fwd_ws_cut']))




