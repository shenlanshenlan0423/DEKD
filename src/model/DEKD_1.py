# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 15:18
@Auth ： Hongwei
@File ：DEKD_1.py
@IDE ：PyCharm
"""
from definitions import *
from src.model.Clutering import run_clustering
from src.utils.model_utils import compute_weight


class DEKD_1:
    def __init__(self, n_clusters, sample_ratio, cols, args):
        self.cols = cols
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.explainer = None
        self.cluster_model = None
        self.classifiers = []
        self.cluster_centers = None
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
        final_cluster_centers = []
        for subclass in range(self.n_clusters):
            sub_shap_array = train_shap_array[np.where(train_label == subclass)[0]]
            point_to_center_distance = self.cluster_model.transform(sub_shap_array)[:, subclass]
            sampled_X_idx = np.argsort(point_to_center_distance)[:int(sub_shap_array.shape[0] * self.sample_ratio)]

            sub_X_train = X_train[np.where(train_label == subclass)[0]][sampled_X_idx]
            # sub_X_shap = sub_shap_array[sampled_X_idx]
            # sub_X_train = np.hstack((sub_X_train, sub_X_shap))
            sub_y_train = y_train[np.where(train_label == subclass)[0]][sampled_X_idx]
            if sub_X_train.shape[0] <= 10:
                print('continue!')
                continue
            # # Note: beeswarm plot for StratificationAnalysis
            # fig = plt.figure(figsize=(8, 6), dpi=200)
            # plt.clf()
            # df_X = pd.DataFrame(sub_X_train, columns=self.cols[:-1])
            # shap.summary_plot(self.explainer(df_X), df_X, show=False, max_display=10)
            # plt.tight_layout()
            # plt.savefig(FIG_DIR + '[cluster-{}]_summary_plot.pdf'.format(subclass))
            # plt.show()

            final_cluster_centers.append(cluster_centers[subclass].tolist())
            base_classifier = XGBClassifier(device=device).fit(sub_X_train, sub_y_train)
            # base_classifier = RandomForestClassifier().fit(sub_X_train, sub_y_train)
            self.classifiers.append(base_classifier)
        self.cluster_centers = np.array(final_cluster_centers)

    def predict_proba(self, X_test):
        # Note: mortality prediction
        X_test = pd.DataFrame(X_test, columns=self.cols[:-1])
        test_shap_array = self.explainer(X_test).values

        distances = np.zeros((X_test.shape[0], self.cluster_centers.shape[0]))
        pred_proba = np.zeros((X_test.shape[0], self.cluster_centers.shape[0]))
        for idx in range(self.cluster_centers.shape[0]):
            center = self.cluster_centers[idx]
            distances[:, idx] = np.linalg.norm(center - test_shap_array, axis=1)
            pred_proba[:, idx] = self.classifiers[idx].predict_proba(X_test)[:, 1]
        weight = compute_weight(distances, self.args.beta)
        y_pred = np.nan_to_num(np.sum(pred_proba * weight, axis=1))
        return np.vstack((1 - y_pred, y_pred)).T
