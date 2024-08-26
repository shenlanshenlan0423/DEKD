# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/27 15:17
@Auth ： Hongwei
@File ：WAUCE_shap.py
@IDE ：PyCharm
"""
from definitions import *
from src.model.Clutering import run_clustering


class WAUCE:
    def __init__(self, n_clusters, sample_ratio, args):
        self.cols = args.cols
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.explainer = None
        self.cluster_model = None
        self.classifiers = []
        self.cluster_centers = []
        self.AUROC = None
        self.args = args

    def fit(self, X_train, y_train):
        X_train = pd.DataFrame(X_train, columns=self.cols[:-1])
        xgb = XGBClassifier(device=device).fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(xgb)
        train_shap_array = self.explainer(X_train).values
        cluster_model, res_dict = run_clustering(train_shap_array, n_clusters=self.n_clusters)
        self.cluster_model = cluster_model['KMeans']
        cluster_centers = self.cluster_model.cluster_centers_
        train_label = self.cluster_model.labels_

        X_train = X_train.values
        for subclass in range(self.n_clusters):
            sub_shap_array = train_shap_array[np.where(train_label == subclass)[0]]
            point_to_center_distance = self.cluster_model.transform(sub_shap_array)[:, subclass]
            sampled_X_idx = np.argsort(point_to_center_distance)[:int(sub_shap_array.shape[0] * self.sample_ratio)]

            sub_X_train = X_train[np.where(train_label == subclass)[0]][sampled_X_idx]
            sub_y_train = y_train[np.where(train_label == subclass)[0]][sampled_X_idx]
            if sub_X_train.shape[0] <= 10:
                print('continue!')
                continue
            self.cluster_centers.append(cluster_centers[subclass].tolist())
            base_classifier = XGBClassifier(device=device).fit(sub_X_train, sub_y_train)
            self.classifiers.append(base_classifier)

    def predict_val(self, X_val, y_val):
        AUROC = []
        for idx in range(len(self.cluster_centers)):
            AUROC.append(roc_auc_score(y_val, self.classifiers[idx].predict(X_val)))
        self.AUROC = np.array(AUROC)

    def predict_proba(self, X_test):
        self.cluster_centers = np.array(self.cluster_centers)
        pred_proba = np.zeros((X_test.shape[0], self.cluster_centers.shape[0]))
        for idx in range(self.cluster_centers.shape[0]):
            pred_proba[:, idx] = self.classifiers[idx].predict_proba(X_test)[:, 1]
        weight = self.AUROC / np.sum(self.AUROC)
        y_pred = np.sum(pred_proba * weight, axis=1)

        return np.vstack((1 - y_pred, y_pred)).T
