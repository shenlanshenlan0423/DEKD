# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/12 11:10
@Auth ： Hongwei
@File ：plot_analyse_for_clusters_and_classifiers.py
@IDE ：PyCharm
"""
from definitions import *

color_dict = {
    0: '#2FAFB5',
    1: '#9DE7DA',
    2: '#F8B9C4',
    3: '#ee8a5f'
}


def plot_principal(fig, X, labels, label_name, subplot_idx):
    ax = fig.add_subplot(2, 2, subplot_idx)
    fontsize = 18
    for label in sorted(pd.Series(labels).unique().tolist()):
        subclass_X = X[np.where(labels == label)]
        ax.scatter(subclass_X[:, 0], subclass_X[:, 1], c=color_dict[label], marker='o', s=20,
                   alpha=1, label='Cluster ' + str(label))
    ax.legend(loc='upper right', fontsize=fontsize)
    if label_name == 'DELAK-Feature':
        ax.set_xlabel('UMAP 1 of raw features\n(a) Raw features stratification result of DELAK', fontsize=fontsize)
        ax.set_ylabel('UMAP 2 of raw features', fontsize=fontsize)
    elif label_name == 'DELAK-SHAP value':
        ax.set_xlabel('UMAP 1 of SHAP values\n(b) SHAP values stratification result of DELAK', fontsize=fontsize)
        ax.set_ylabel('UMAP 2 of SHAP values', fontsize=fontsize)
    elif label_name == 'DEKD (Stage 1)-Feature':
        ax.set_xlabel('UMAP 1 of raw features\n(c) Raw features stratification result of DEKD (Stage 1)', fontsize=fontsize)
        ax.set_ylabel('UMAP 2 of raw features', fontsize=fontsize)
    elif label_name == 'DEKD (Stage 1)-SHAP value':
        ax.set_xlabel('UMAP 1 of SHAP values\n(d) SHAP values stratification result of DEKD (Stage 1)', fontsize=fontsize)
        ax.set_ylabel('UMAP 2 of SHAP values', fontsize=fontsize)


def plot_StratificationPredict(fig, X, probas, model, subplot_idx):
    ax = fig.add_subplot(1, 2, subplot_idx)
    fontsize = 24
    scatter = plt.scatter(X[:, 0], X[:, 1], c=probas, cmap='GnBu', s=4, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.45, aspect=8, pad=0.06)
    cbar.ax.tick_params(labelsize=15)

    if model == 'DELAK':
        ax.text(14.5, 8.75, "Prediction\nProbability", fontsize=fontsize)
        ax.set_xlabel('UMAP 1 of raw features\n(a) Prediction probability by DELAK', fontsize=fontsize)
        ax.set_ylabel('UMAP 2 of raw features', fontsize=fontsize)
    elif model == 'DEKD (Stage 1)':
        ax.text(19.5, 15.5, "Prediction\nProbability", fontsize=fontsize)
        ax.set_xlabel('UMAP 1 of SHAP values\n(b) Prediction probability by DEKD (Stage 1)', fontsize=fontsize)
        ax.set_ylabel('UMAP 2 of SHAP values', fontsize=fontsize)


def plot_SHAP_summary_plot(X, cols, model):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    df_X = pd.DataFrame(X, columns=cols[:-1])
    explainer = shap.TreeExplainer(model.xgb)
    shap.summary_plot(explainer(df_X), df_X, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'DEKD_summary_plot.pdf')
    plt.show()

    fig = plt.figure(figsize=(8, 6), dpi=200)
    shap.summary_plot(explainer(df_X), df_X, plot_type="bar", show=False, max_display=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR + 'DEKD_bar_summary_plot.pdf')
    plt.show()


def get_metric_res(res_dicts):
    DM, CC, Q, Kappa = [], [], [], []
    for res_dict in res_dicts:
        DM.append(res_dict['Disagreement measure'])
        CC.append(res_dict['Matthews Corrcoef'])
        Q.append(res_dict['Q-statistic'])
        Kappa.append(res_dict['Kappa-statistic'])
    return np.array([DM, CC, Kappa])


def plot_DiversityMeasure(res_dicts):
    rf_arr = get_metric_res(res_dicts['RF'])
    knorau_arr = get_metric_res(res_dicts['KNORAU'])
    delak_arr = get_metric_res(res_dicts['DELAK'])
    dekd_1_arr = get_metric_res(res_dicts['DEKD (Stage 1)'])

    metric_list = ['DM', 'CC', 'Kappa']

    fig = plt.figure(figsize=(10, 5), dpi=200)
    plt.errorbar([0, 4, 8], rf_arr.T.mean(axis=0), yerr=rf_arr.T.std(axis=0),
                 fmt='o', capsize=6, color='black', linewidth=2, markersize=8, label='RF')
    plt.errorbar([1, 5, 9], knorau_arr.T.mean(axis=0), yerr=knorau_arr.T.std(axis=0),
                 fmt='d', capsize=6, color='black', linewidth=2, markersize=8, label='KNORAU')
    plt.errorbar([2, 6, 10], delak_arr.T.mean(axis=0), yerr=delak_arr.T.std(axis=0),
                 fmt='p', capsize=6, color='black', linewidth=2, markersize=8, label='DELAK')
    plt.errorbar([3, 7, 11], dekd_1_arr.T.mean(axis=0), yerr=dekd_1_arr.T.std(axis=0),
                 fmt='x', capsize=6, color='teal', linewidth=2, markersize=10, label='DEKD (Stage 1)')

    plt.axvline(x=3.5, color='black', linestyle='dashed')
    plt.axvline(x=7.5, color='black', linestyle='dashed')

    plt.xticks([1.5, 5.5, 9.5], metric_list, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1.05])
    plt.xlabel('Metrics', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.grid(axis='both')

    plt.tight_layout()
    plt.savefig(FIG_DIR + 'DiversityMeasure.pdf')
    plt.show()
