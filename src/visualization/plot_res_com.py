# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/28 10:00
@Auth ： Hongwei
@File ：plot_res_com.py
@IDE ：PyCharm
"""
from definitions import *


def plot_part1_bar(ml_res, myModel_res, fig, idx):
    metrics = ['AUROC', 'AUPRC']
    y_values = {'AUROC': [], 'AUPRC': []}
    ml_model_names = list(ml_res[0].keys())
    for i, model_name in enumerate(ml_model_names):
        for j, metric in enumerate(metrics):
            metric_values = [ml_res[idx_][model_name][metric] for idx_ in range(len(ml_res))]
            y_values[metric].append(metric_values)
    myModel_names = ['DEKD (Stage 1)', 'DEKD']
    for i, model_name in enumerate(myModel_names):
        for j, metric in enumerate(metrics):
            metric_values = [myModel_res[idx_][model_name][metric] for idx_ in range(len(myModel_res))]
            y_values[metric].append(metric_values)

    model_names = ['LR', 'DT', 'Bagging', 'RF', 'Adaboost', 'XGBoost', 'MLP', 'DEKD (Stage 1)', 'DEKD']

    ax = fig.add_subplot(6, 2, idx+1)
    metric_values_table = np.array(y_values['AUROC']).T
    for index, model in enumerate(model_names):
        ax.bar(index, metric_values_table.mean(axis=0)[index], yerr=metric_values_table.std(axis=0)[index],
               capsize=6, color='#F39DA3', linewidth=2, alpha=0.6)
    ax.set_xticks(range(len(model_names)), model_names, fontsize=12, rotation=30)
    ax.set_ylabel('AUROC', fontsize=15)
    if idx == 0:
        ax.set_ylim([0.73, 0.92])
    elif idx == 1:
        ax.set_ylim([0.7, 0.86])
    ax.grid(axis='y', linestyle='dashed')

    ax = fig.add_subplot(6, 2, idx+3)
    metric_values_table = np.array(y_values['AUPRC']).T
    for index, model in enumerate(model_names):
        ax.bar(index, metric_values_table.mean(axis=0)[index], yerr=metric_values_table.std(axis=0)[index],
               capsize=6, color='#34C3B7', linewidth=2, alpha=0.6)
    ax.set_xticks(range(len(model_names)), model_names, fontsize=12, rotation=30)
    ax.set_ylabel('AUPRC', fontsize=15)
    if idx == 0:
        ax.set_ylim([0.68, 1])
        ax.set_xlabel('(a)', labelpad=12, fontsize=24)
    elif idx == 1:
        ax.set_ylim([0.6, 0.93])
        ax.set_xlabel('(b)', labelpad=12, fontsize=24)
    ax.grid(axis='y', linestyle='dashed')


def plot_part2_bar(ml_res, myModel_res, fig, idx):
    metrics = ['AUROC', 'AUPRC']
    y_values = {'AUROC': [], 'AUPRC': []}
    ml_model_names = list(ml_res[0].keys())
    for i, model_name in enumerate(ml_model_names):
        for j, metric in enumerate(metrics):
            metric_values = [ml_res[idx_][model_name][metric] for idx_ in range(len(ml_res))]
            y_values[metric].append(metric_values)
    myModel_names = ['DEKD (Stage 1)', 'DEKD']
    for i, model_name in enumerate(myModel_names):
        for j, metric in enumerate(metrics):
            metric_values = [myModel_res[idx_][model_name][metric] for idx_ in range(len(myModel_res))]
            y_values[metric].append(metric_values)

    model_names = ['AvgE', 'MaxE', 'MinE', 'WAUCE', 'SB', 'DEKD (Stage 1)', 'DEKD']

    ax = fig.add_subplot(6, 2, idx+5)
    metric_values_table = np.array(y_values['AUROC']).T
    for index, model in enumerate(model_names):
        ax.bar(index, metric_values_table.mean(axis=0)[index], yerr=metric_values_table.std(axis=0)[index],
               capsize=6, color='#F39DA3', linewidth=2, alpha=0.6)
    ax.set_xticks(range(len(model_names)), model_names, fontsize=12, rotation=15)
    ax.set_ylabel('AUROC', fontsize=15)
    if idx == 0:
        # ax.set_ylim([0.65, 0.9])
        ax.set_ylim([0.5, 0.9])
    elif idx == 1:
        # ax.set_ylim([0.65, 0.86])
        ax.set_ylim([0.5, 0.86])
    ax.grid(axis='y', linestyle='dashed')

    ax = fig.add_subplot(6, 2, idx+7)
    metric_values_table = np.array(y_values['AUPRC']).T
    for index, model in enumerate(model_names):
        ax.bar(index, metric_values_table.mean(axis=0)[index], yerr=metric_values_table.std(axis=0)[index],
               capsize=6, color='#34C3B7', linewidth=2, alpha=0.6)
    ax.set_xticks(range(len(model_names)), model_names, fontsize=12, rotation=15)
    ax.set_ylabel('AUPRC', fontsize=15)
    if idx == 0:
        # ax.set_ylim([0.82, 0.97])
        ax.set_ylim([0.6, 0.97])
        ax.set_xlabel('(c)', labelpad=12, fontsize=24)
    elif idx == 1:
        # ax.set_ylim([0.79, 0.93])
        ax.set_ylim([0.48, 0.93])
        ax.set_xlabel('(d)', labelpad=12, fontsize=24)
    ax.grid(axis='y', linestyle='dashed')


def plot_part3_bar(ml_res, myModel_res, fig, idx):
    metrics = ['AUROC', 'AUPRC']
    y_values = {'AUROC': [], 'AUPRC': []}
    ml_model_names = list(ml_res[0].keys())
    for i, model_name in enumerate(ml_model_names):
        for j, metric in enumerate(metrics):
            metric_values = [ml_res[idx_][model_name][metric] for idx_ in range(len(ml_res))]
            y_values[metric].append(metric_values)
    myModel_names = ['DELAK', 'DEKD (Stage 1)', 'DEKD']
    for i, model_name in enumerate(myModel_names):
        for j, metric in enumerate(metrics):
            metric_values = [myModel_res[idx_][model_name][metric] for idx_ in range(len(myModel_res))]
            y_values[metric].append(metric_values)

    model_names = ['DESP', 'KNORAU', 'KNORAE', 'METADES', 'DELAK', 'DEKD (Stage 1)', 'DEKD']

    ax = fig.add_subplot(6, 2, idx+9)
    metric_values_table = np.array(y_values['AUROC']).T
    for index, model in enumerate(model_names):
        ax.bar(index, metric_values_table.mean(axis=0)[index], yerr=metric_values_table.std(axis=0)[index],
               capsize=6, color='#F39DA3', linewidth=2, alpha=0.6)
    ax.set_xticks(range(len(model_names)), model_names, fontsize=12, rotation=15)
    ax.set_ylabel('AUROC', fontsize=15)
    if idx == 0:
        ax.set_ylim([0.82, 0.9])
    elif idx == 1:
        ax.set_ylim([0.7, 0.86])
    ax.grid(axis='y', linestyle='dashed')

    ax = fig.add_subplot(6, 2, idx+11)
    metric_values_table = np.array(y_values['AUPRC']).T
    for index, model in enumerate(model_names):
        ax.bar(index, metric_values_table.mean(axis=0)[index], yerr=metric_values_table.std(axis=0)[index],
               capsize=6, color='#34C3B7', linewidth=2, alpha=0.6)
    ax.set_xticks(range(len(model_names)), model_names, fontsize=12, rotation=15)
    ax.set_ylabel('AUPRC', fontsize=15)
    if idx == 0:
        ax.set_ylim([0.9, 0.96])
        ax.set_xlabel('(e)\n48 hours mortality prediction', labelpad=12, fontsize=24)
    elif idx == 1:
        ax.set_ylim([0.85, 0.93])
        ax.set_xlabel('(f)\nIn-hospital mortality prediction', labelpad=12, fontsize=24)
    ax.grid(axis='y', linestyle='dashed')


if __name__ == '__main__':
    for balance_ratio in [1]:
        fig = plt.figure(figsize=(15, 20), dpi=200)
        for idx, outcome_label in enumerate(['48h_mortality', 'hospital_mortality']):
            part1_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_part1.pickle'.format(outcome_label, balance_ratio))
            part2_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_part2.pickle'.format(outcome_label, balance_ratio))
            part3_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_part3.pickle'.format(outcome_label, balance_ratio))
            myModels_res = load_pickle(
                RESULT_DIR + '{}/[BR-1-{}]_eval_res_myModel.pickle'.format(outcome_label, balance_ratio))
            plot_part1_bar(part1_res, myModels_res, fig=fig, idx=idx)
            plot_part2_bar(part2_res, myModels_res, fig=fig, idx=idx)
            plot_part3_bar(part3_res, myModels_res, fig=fig, idx=idx)
        plt.tight_layout()
        plt.savefig(FIG_DIR + '/res_com_errorbar.pdf')
        plt.show()
