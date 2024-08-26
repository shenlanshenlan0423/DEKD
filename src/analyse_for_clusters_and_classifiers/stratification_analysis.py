# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/28 11:39
@Auth ： Hongwei
@File ：analyse_for_clusters_and_classifiers.py
@IDE ：PyCharm
"""
from definitions import *
import umap
from src.visualization.plot_analyse_for_clusters_and_classifiers import plot_principal, plot_StratificationPredict, plot_SHAP_summary_plot


def get_StratificationRes(X, DELAK, DEKD_1):
    fig = plt.figure(figsize=(15, 12), dpi=200)
    umap_model = umap.UMAP(random_state=0)
    save_pickle(umap_model.fit_transform(X), RESULT_DIR+'DELAK_X.pickle')
    save_pickle(umap_model.fit_transform(DEKD_1.explainer(X).values), RESULT_DIR + 'DEKD_1_X.pickle')

    plot_principal(fig, load_pickle(RESULT_DIR+'DELAK_X.pickle'), DELAK.cluster_model.labels_, 'DELAK-Feature', 1)
    plot_principal(fig, load_pickle(RESULT_DIR+'DEKD_1_X.pickle'), DELAK.cluster_model.labels_, 'DELAK-SHAP value', 2)
    plot_principal(fig, load_pickle(RESULT_DIR+'DELAK_X.pickle'), DEKD_1.cluster_model.labels_, 'DEKD (Stage 1)-Feature', 3)
    plot_principal(fig, load_pickle(RESULT_DIR+'DEKD_1_X.pickle'), DEKD_1.cluster_model.labels_, 'DEKD (Stage 1)-SHAP value', 4)
    plt.tight_layout()
    plt.savefig(FIG_DIR + '/StratificationRes.pdf')
    plt.show()


def get_StratificationPredictRes(X, DELAK, SHAPDE):
    fig = plt.figure(figsize=(23, 8), dpi=200)
    plot_StratificationPredict(fig, load_pickle(RESULT_DIR+'DELAK_X.pickle'), DELAK.predict_proba(X)[:, 0], 'DELAK', 1)
    plot_StratificationPredict(fig, load_pickle(RESULT_DIR+'DEKD_1_X.pickle'), SHAPDE.predict_proba(X)[:, 0], 'DEKD (Stage 1)', 2)
    plt.tight_layout()
    plt.savefig(FIG_DIR + '/StratificationPredictRes.pdf')
    plt.show()


if __name__ == '__main__':
    for outcome_label in ['48h_mortality']:
        for balance_ratio in [1]:
            dataset_path = DATA_DIR + '/processed_data/{}/{}_folds_datasets_BR-[1-{}].pickle'.format(
                outcome_label,
                fold_number,
                balance_ratio)
            fold_data = load_pickle(dataset_path)['fold 2']
            cols, train_set, val_set, test_set = fold_data['cols'], fold_data['train'], fold_data['val'], fold_data['test']
            X_train, X_val, X_test = train_set[:, :-1], val_set[:, :-1], test_set[:, :-1]
            y_train, y_val, y_test = train_set[:, -1], val_set[:, -1], test_set[:, -1]

            trained_model_path = MODELS_DIR + '/{}/'.format(outcome_label) + '[BR-1-{}]_myModel.pickle'.format(
                balance_ratio)
            models = load_pickle(trained_model_path)['fold 2']
            DELAK, DEKD_1 = models['DELAK'], models['DEKD (Stage 1)']
            get_StratificationRes(X_train, DELAK, DEKD_1)
            get_StratificationPredictRes(X_train, DELAK, DEKD_1)
            plot_SHAP_summary_plot(X_train, cols, model=models['DEKD'])
