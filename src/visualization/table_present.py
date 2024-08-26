# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/14 12:02
@Auth ： Hongwei
@File ：table_present.py
@IDE ：PyCharm
"""
from definitions import *
from src.utils.main_uitls import dict_concat, get_results


def mark_res(df):
    cols = df.columns.tolist()[1:]
    for col_idx in range(len(cols)):
        col_strings = df.iloc[:, col_idx + 1].tolist()
        col_value = np.array([float(i[:6]) for i in col_strings])
        rank_list = np.zeros_like(col_value)
        for i, idx in enumerate(np.argsort(-col_value)):
            rank_list[idx] = i + 1
        df_series = pd.Series(rank_list.astype(int), name='Rank')
        df = pd.concat([df, df_series], axis=1)
    return df.iloc[:, [0, 1, 3, 2, 4]]


def part1_present(task_res, filepath):
    part_table = pd.DataFrame()
    part_table['Models'] = ['LR', 'DT', 'Bagging', 'RF', 'Adaboost', 'XGBoost', 'MLP', 'DEKD (Stage 1)', 'DEKD']
    for res_df in task_res:
        part_table = pd.concat([part_table, mark_res(res_df).iloc[:, 1:]], axis=1)
    part_table.to_excel(filepath, index=False)


def part2_present(task_res, filepath):
    part_table = pd.DataFrame()
    part_table['Models'] = ['AvgE', 'MaxE', 'MinE', 'WAUCE', 'SB', 'DEKD (Stage 1)', 'DEKD']
    for res_df in task_res:
        part_table = pd.concat([part_table, mark_res(res_df).iloc[:, 1:]], axis=1)
    part_table.to_excel(filepath, index=False)


def part3_present(task_res, filepath):
    part_table = pd.DataFrame()
    part_table['Models'] = ['DESP', 'KNORAU', 'KNORAE', 'METADES', 'DELAK', 'DEKD (Stage 1)', 'DEKD']
    for res_df in task_res:
        part_table = pd.concat([part_table, mark_res(res_df).iloc[:, 1:]], axis=1)
    part_table.to_excel(filepath, index=False)


if __name__ == '__main__':
    for balance_ratio in [1]:
        task_res = []
        for idx, outcome_label in enumerate(['48h_mortality', 'hospital_mortality']):
            part1_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_part1.pickle'.format(outcome_label, balance_ratio))
            myModels_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_myModel.pickle'.format(outcome_label, balance_ratio))
            task_res.append(get_results(dict_concat(part1_res, myModels_res), ['LR', 'DT', 'BG', 'RF', 'ADA', 'XGB', 'MLP', 'DEKD (Stage 1)', 'DEKD']))
        part1_present(task_res, filepath=TABLE_DIR+'BR-[1-{}]_with_ml_comparison.xlsx'.format(balance_ratio))


        task_res = []
        for idx, outcome_label in enumerate(['48h_mortality', 'hospital_mortality']):
            part2_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_part2.pickle'.format(outcome_label, balance_ratio))
            myModels_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_myModel.pickle'.format(outcome_label, balance_ratio))
            task_res.append(get_results(dict_concat(part2_res, myModels_res), ['AvgE', 'MaxE', 'MinE', 'WAUCE', 'SB', 'DEKD (Stage 1)', 'DEKD']))
        part2_present(task_res, filepath=TABLE_DIR+'BR-[1-{}]_with_other_fusion_comparison.xlsx'.format(balance_ratio))


        task_res = []
        for idx, outcome_label in enumerate(['48h_mortality', 'hospital_mortality']):
            part3_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_part3.pickle'.format(outcome_label, balance_ratio))
            myModels_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_myModel.pickle'.format(outcome_label, balance_ratio))
            task_res.append(get_results(dict_concat(part3_res, myModels_res), ['DESP', 'KNORAU', 'KNORAE', 'METADES', 'DELAK', 'DEKD (Stage 1)', 'DEKD']))
        part3_present(task_res, filepath=TABLE_DIR+'BR-[1-{}]_with_SOTA_comparison.xlsx'.format(balance_ratio))
