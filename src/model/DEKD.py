# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 15:18
@Auth ： Hongwei
@File ：DEKD.py
@IDE ：PyCharm
"""
from definitions import *
from src.model.Clutering import run_clustering
from src.utils.model_utils import compute_weight


class DEKD:
    def __init__(self, n_clusters, sample_ratio, cols, args, idx):
        self.cols = cols
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.explainer = None
        self.classifiers = None
        self.cluster_centers = None
        self.args = args
        self.idx = idx
        self.xgb = None
        self.y_proba_by_teacher = None

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = pd.DataFrame(X_train, columns=self.cols[:-1])
        X_val = pd.DataFrame(X_val, columns=self.cols[:-1])
        xgb = XGBClassifier(device=device).fit(X_train, y_train)
        self.explainer = shap.TreeExplainer(xgb)
        self.iterative_fit(X_train, y_train)
        self.y_proba_by_teacher = self.teacher_predict_proba(X_train)
        self.iterative_kd(X_train, y_train, X_val, y_val)

    def iterative_fit(self, X_train, y_train):
        train_shap_array = self.explainer(X_train).values
        cluster_model, res_dict = run_clustering(train_shap_array, n_clusters=self.n_clusters)
        cluster_model = cluster_model['KMeans']
        cluster_centers = cluster_model.cluster_centers_
        train_label = cluster_model.labels_

        X_train = X_train.values
        classifiers, final_cluster_centers = [], []
        for subclass in range(self.n_clusters):
            sub_shap_array = train_shap_array[np.where(train_label == subclass)[0]]
            point_to_center_distance = cluster_model.transform(sub_shap_array)[:, subclass]
            sampled_X_idx = np.argsort(point_to_center_distance)[:int(sub_shap_array.shape[0] * self.sample_ratio)]

            sub_X_train = X_train[np.where(train_label == subclass)[0]][sampled_X_idx]
            sub_y_train = y_train[np.where(train_label == subclass)[0]][sampled_X_idx]
            if sub_X_train.shape[0] <= 10:
                print('continue!')
                continue
            final_cluster_centers.append(cluster_centers[subclass].tolist())
            base_classifier = XGBClassifier(device=device).fit(sub_X_train, sub_y_train)
            # base_classifier = RandomForestClassifier().fit(sub_X_train, sub_y_train)
            classifiers.append(base_classifier)
        self.classifiers = classifiers
        self.cluster_centers = np.array(final_cluster_centers)

    def teacher_predict_proba(self, X):
        # mortality prediction
        X = pd.DataFrame(X, columns=self.cols[:-1])
        shap_array = self.explainer(X).values

        distances = np.zeros((X.shape[0], self.cluster_centers.shape[0]))
        pred_proba = np.zeros((X.shape[0], self.cluster_centers.shape[0]))
        for idx in range(self.cluster_centers.shape[0]):
            center = self.cluster_centers[idx]
            distances[:, idx] = np.linalg.norm(center - shap_array, axis=1)
            pred_proba[:, idx] = self.classifiers[idx].predict_proba(X)[:, 1]
        weight = compute_weight(distances, self.args.beta)
        y_pred = np.nan_to_num(np.sum(pred_proba * weight, axis=1))
        return y_pred

    def AUROC(self, preds, dtrain):
        y = dtrain.get_label()
        AUROC = roc_auc_score(y, preds)
        return 'AUROC', AUROC

    def binary_cross_entropy(self, pred, dtrain):
        T, alpha = self.args.Temperature, self.args.Alpha
        label = self.y_proba_by_teacher
        true_label = dtrain.get_label()

        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        label_logits = np.log(label / (1 - label))
        label = np.exp(label_logits / T) / (1 + np.exp(label_logits / T))

        # soft grad & hess
        soft_grad = -(1 ** label) * (label - sigmoid_pred)
        soft_hess = (1 ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

        # hard grad & hess
        hard_grad = -(1 ** true_label) * (true_label - sigmoid_pred)
        hard_hess = (1 ** true_label) * sigmoid_pred * (1.0 - sigmoid_pred)

        grad = (1 - alpha) * soft_grad + alpha * hard_grad
        hess = (1 - alpha) * soft_hess + alpha * hard_hess

        return grad, hess

    def iterative_kd(self, X_train, y_train, X_val, y_val):
        params = {
            'silent': 1,
            'objective': 'binary:logistic',
            'gamma': 0.1,
            'min_child_weight': 5,
            'disable_default_eval_metric': 1,
        }

        d_train = xgboost.DMatrix(X_train, y_train)
        d_val = xgboost.DMatrix(X_val, y_val)
        self.xgb = xgboost.train(params=params,
                                 dtrain=d_train,
                                 num_boost_round=self.args.num_boost_round,
                                 custom_metric=self.AUROC,
                                 evals=[(d_train, 'dtrain'), (d_val, 'dval')],
                                 obj=self.binary_cross_entropy,
                                 verbose_eval=False)

    def predict_proba(self, X_test):
        X_test = pd.DataFrame(X_test, columns=self.cols[:-1])
        d_test = xgboost.DMatrix(X_test)
        y_pred = self.xgb.predict(d_test)  # mortality
        return np.vstack((1 - y_pred, y_pred)).T
