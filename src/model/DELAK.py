# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 20:10
@Auth ： Hongwei
@File ：DELAK.py
@IDE ：PyCharm
"""
from definitions import *
from src.model.Clutering import run_clustering
from src.utils.model_utils import compute_weight


class DELAK:
    def __init__(self, n_clusters, sample_ratio, args):
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.cluster_model = None
        self.classifiers = []
        self.cluster_centers = None
        self.args = args

    def fit(self, X_train, y_train):
        cluster_model, res_dict = run_clustering(X_train, n_clusters=self.n_clusters)
        self.cluster_model = cluster_model['KMeans']
        cluster_centers = self.cluster_model.cluster_centers_
        train_label = self.cluster_model.labels_

        centers = []
        for subclass in range(self.n_clusters):
            sub_X_train = X_train[np.where(train_label == subclass)[0]]
            sub_y_train = y_train[np.where(train_label == subclass)[0]]

            point_to_center_distance = self.cluster_model.transform(sub_X_train)[:, subclass]
            sampled_X_idx = np.argsort(point_to_center_distance)[:int(sub_X_train.shape[0] * self.sample_ratio)]
            sub_X_train, sub_y_train = sub_X_train[sampled_X_idx], sub_y_train[sampled_X_idx]
            if sub_X_train.shape[0] <= 10:
                print('continue!')
                continue
            centers.append(cluster_centers[subclass].tolist())
            base_classifier = XGBClassifier(device=device).fit(sub_X_train, sub_y_train)
            # base_classifier = RandomForestClassifier().fit(sub_X_train, sub_y_train)
            self.classifiers.append(base_classifier)
        self.cluster_centers = np.array(centers)

    def predict_proba(self, X_test):
        distances = np.zeros((X_test.shape[0], self.cluster_centers.shape[0]))
        pred_proba = np.zeros((X_test.shape[0], self.cluster_centers.shape[0]))
        for idx in range(self.cluster_centers.shape[0]):
            center = self.cluster_centers[idx]
            distances[:, idx] = np.linalg.norm(center - X_test, axis=1)
            pred_proba[:, idx] = self.classifiers[idx].predict_proba(X_test)[:, 1]
        weight = compute_weight(distances, self.args.beta)
        y_pred = np.nan_to_num(np.sum(pred_proba * weight, axis=1))
        return np.vstack((1 - y_pred, y_pred)).T
