# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/22 13:45
@Auth ： Hongwei
@File ：diversity_measure.py
@IDE ：PyCharm
"""
from definitions import *
import itertools
from src.visualization.plot_analyse_for_clusters_and_classifiers import plot_DiversityMeasure
from sklearn.metrics import matthews_corrcoef


def classifierTest(fold_data, classifiers):
    evaluation_results = []
    cols, val_set, test_set = fold_data['cols'], fold_data['val'], fold_data['test']
    X_val, X_test = val_set[:, :-1], test_set[:, :-1]

    pairs = list(itertools.combinations(classifiers, 2))
    for idx, pair in enumerate(pairs):
        y_1 = pair[0].predict(X_test)
        y_2 = pair[1].predict(X_test)
        evaluation_results.append(get_DiversityMeasure(y_1, y_2))
    return evaluation_results


def get_DiversityMeasure(y_1, y_2):
    d, b, c, a = confusion_matrix(y_1, y_2).ravel()  # tn, fp, fn, tp
    dis = (b + c) / len(y_1)
    pho = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (c + d) * (b + d))
    Q = (a * d - b * c) / (a * d + b * c)

    p1 = (a + d) / len(y_1)
    p2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (len(y_1) ** 2)
    kappa = (p1 - p2) / (1 - p2)

    res_dict = {
        'Disagreement measure': dis,
        'Matthews Corrcoef': matthews_corrcoef(y_1, y_2),
        'Q-statistic': Q,
        'Kappa-statistic': kappa
    }
    return res_dict


if __name__ == '__main__':
    for outcome_label in ['48h_mortality']:
        for balance_ratio in [1]:
            dataset_path = DATA_DIR + '/processed_data/{}/{}_folds_datasets_BR-[1-{}].pickle'.format(
                outcome_label,
                fold_number,
                balance_ratio)
            MyModels_path = MODELS_DIR + '/{}/'.format(outcome_label) + '[BR-1-{}]_myModel.pickle'.format(
                balance_ratio)
            Part1_models_path = MODELS_DIR + '/{}/'.format(outcome_label) + '[BR-1-{}]_part1.pickle'.format(
                balance_ratio)
            Part3_models_path = MODELS_DIR + '/{}/'.format(outcome_label) + '[BR-1-{}]_part3.pickle'.format(
                balance_ratio)

            dataset = load_pickle(dataset_path)
            MyModels, Part1_models, Part3_models = load_pickle(MyModels_path), load_pickle(Part1_models_path), load_pickle(Part3_models_path)
            total_res = {'DELAK': [], 'DEKD (Stage 1)': [], 'RF': [], 'KNORAU': []}
            for idx, fold_idx in enumerate(tqdm(list(dataset.keys()))):
                fold_data = dataset[fold_idx]
                DELAK, DEKD_1 = MyModels[fold_idx]['DELAK'], MyModels[fold_idx]['DEKD_1']
                RF, KNORAU = Part1_models[fold_idx]['RF'], Part3_models[fold_idx]['KNORAU']
                total_res['DELAK'].extend(classifierTest(fold_data, DELAK.classifiers))
                total_res['DEKD (Stage 1)'].extend(classifierTest(fold_data, DEKD_1.classifiers))
                total_res['RF'].extend(classifierTest(fold_data, RF.estimators_))
                total_res['KNORAU'].extend(classifierTest(fold_data, KNORAU.pool_classifiers.estimators_))
            # save_pickle(total_res, RESULT_DIR+'diversity_measure_res.pickle')
            # total_res = load_pickle(RESULT_DIR+'diversity_measure_res.pickle')
            plot_DiversityMeasure(total_res)
