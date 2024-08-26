# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/20 21:49
@Auth ： Hongwei
@File ：Clutering.py
@IDE ：PyCharm
"""
from sklearn.cluster import KMeans
from sklearn import preprocessing
from definitions import *
from sklearn import metrics


def train_clustering(model_name, X, n_clusters=None):
    if model_name in ['KMeans']:
        model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(X)
    return model


def print_results(evaluation_results, idx):
    print('--------------【Res. of subset:[{}-{}]】--------------'.format(idx[0], idx[1]))
    model_names = list(evaluation_results[0].keys())
    metrics = ['Silhouette coefficient', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
    for model_name in model_names:
        print('\t【Res. of {}】'.format(model_name))
        for metric in metrics:
            metric_values = [evaluation_results[idx_][model_name][metric] for idx_ in
                             range(len(evaluation_results))]
            print('\t\t{}: {:.4f}'.format(metric, np.mean(metric_values)))


def model_evaluate(model, X):
    labels = model.labels_
    res_dict = {
        'Silhouette coefficient': metrics.silhouette_score(X, labels),
        'Calinski-Harabasz Index': metrics.calinski_harabasz_score(X, labels),
        'Davies-Bouldin Index': metrics.davies_bouldin_score(X, labels)
    }
    return res_dict


def run_clustering(X_train, n_clusters, fold_idx=None, load_model=False):
    model_dir = MODELS_DIR
    evaluation_results = []
    if not load_model:
        kmeans = train_clustering(model_name='KMeans', X=X_train.astype(float), n_clusters=n_clusters)
        trained_model_dict = {
            'KMeans': kmeans,
        }
    else:
        trained_model_dict = load_pickle(model_dir + '/cluster_model.pickle')[fold_idx]
        kmeans = trained_model_dict['KMeans']
    evaluation_result = {
        'KMeans': model_evaluate(model=kmeans, X=X_train),
    }
    evaluation_results.append(evaluation_result)
    return trained_model_dict, evaluation_results