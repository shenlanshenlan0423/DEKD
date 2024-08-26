# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 11:48
@Auth ： Hongwei
@File ：main_uitls.py
@IDE ：PyCharm
"""
from definitions import *
from src.evaluate.evaluate import paired_t_test
from src.model.AvgE_shap import AvgE
from src.model.MaxE_shap import MaxE
from src.model.MinE_shap import MinE
from src.model.WAUCE_shap import WAUCE
from src.model.SB_shap import SB
from src.model.DELAK import DELAK
from src.model.DEKD_1 import DEKD_1
from src.model.DEKD import DEKD


def train_model(model_name, X_train, y_train, X_val=None, y_val=None, cols=None, args=None, idx=None):
    if model_name == 'LR':
        model = LogisticRegression()

    elif model_name == 'DT':
        model = DecisionTreeClassifier()

    elif model_name == 'BG':
        model = BaggingClassifier()

    elif model_name == 'RF':
        model = RandomForestClassifier()

    elif model_name == 'XGB':
        model = XGBClassifier(device=device)

    elif model_name == 'CAT':
        model = CatBoostClassifier(verbose=False)  # do not print the training progress

    elif model_name == 'ADA':
        model = AdaBoostClassifier()

    elif model_name == 'MLP':
        model = MLPClassifier()

    elif model_name == 'AvgE':
        model = AvgE(n_clusters=args.n_clusters, sample_ratio=1, args=args)

    elif model_name == 'MaxE':
        model = MaxE(n_clusters=args.n_clusters, sample_ratio=1, args=args)

    elif model_name == 'MinE':
        model = MinE(n_clusters=args.n_clusters, sample_ratio=1, args=args)

    elif model_name == 'WAUCE':
        model = WAUCE(n_clusters=args.n_clusters, sample_ratio=1, args=args)

    elif model_name == 'SB':
        model = SB(n_clusters=args.n_clusters, sample_ratio=1, args=args)

    elif model_name == 'DESP':
        # As a base classifier for pool, RF has the best performance
        pool_classifiers = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
        model = DESP(pool_classifiers, random_state=0)

    elif model_name == 'KNORAU':
        pool_classifiers = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
        model = KNORAU(pool_classifiers, random_state=0)

    elif model_name == 'KNORAE':
        pool_classifiers = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
        model = KNORAE(pool_classifiers, random_state=0)

    elif model_name == 'METADES':
        pool_classifiers = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
        model = METADES(pool_classifiers, random_state=0)

    # Our models
    elif model_name == 'DELAK':
        model = DELAK(n_clusters=3, sample_ratio=0.99, args=args)

    elif model_name == 'DEKD (Stage 1)':
        model = DEKD_1(n_clusters=args.n_clusters, sample_ratio=1, cols=cols, args=args)

    elif model_name == 'DEKD':
        model = DEKD(n_clusters=args.n_clusters, sample_ratio=0.96, cols=cols, args=args, idx=idx)

    if model_name in ['DEKD']:
        model.fit(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train)

    return model


def get_results(evaluation_results, model_names):
    metrics = ['AUROC', 'AUPRC']
    result_dataframe = pd.DataFrame(columns=['Models'] + metrics)
    result_dataframe['Models'] = model_names
    for i, model_name in enumerate(model_names):
        for j, metric in enumerate(metrics):
            proposed_model_values = [evaluation_results[idx_][model_names[-1]][metric] for idx_ in
                                     range(len(evaluation_results))]
            metric_values = [evaluation_results[idx_][model_name][metric] for idx_ in
                             range(len(evaluation_results))]
            result_string = paired_t_test(proposed_model_values, metric_values)
            result_dataframe.iloc[i, j + 1] = result_string
    return result_dataframe


def dict_concat(baseline_dict, myModel_dict):
    """
    concatenate the baseline eval res and my model eval res
    """
    mymodel_model_names = list(myModel_dict[0].keys())
    res_dict = copy.deepcopy(baseline_dict)
    for fold_idx in range(fold_number):
        for model in mymodel_model_names:
            res_dict[fold_idx][model] = myModel_dict[fold_idx][model]
    return res_dict
