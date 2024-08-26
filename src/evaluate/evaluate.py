# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 9:22
@Auth ： Hongwei
@File ：evaluate.py
@IDE ：PyCharm
"""
from definitions import *


def model_evaluate(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_label = [1 if i >= 0.5 else 0 for i in y_pred_proba]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_label).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    res_dict = {
        'Accuracy': accuracy_score(y_test, y_pred_label),
        'Precision': precision_score(y_test, y_pred_label),
        'Recall': recall_score(y_test, y_pred_label),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'FNR': fnr,
        'FPR': fpr,
        'F1': f1_score(y_test, y_pred_label),
        'AUROC': roc_auc_score(y_test, y_pred_label),
        'AUPRC': average_precision_score(y_test, y_pred_proba)
    }
    return res_dict


def star_by_p_value(p_value, String):
    if float(p_value) < 0.001:
        newString = String + '***'
    elif float(p_value) < 0.05:
        newString = String + '**'
    elif float(p_value) < 0.01:
        newString = String + '*'
    else:
        newString = String
    return newString


def paired_t_test(proposed_model_values, other_model_values):
    t_statistic, p_value = stats.ttest_ind(proposed_model_values, other_model_values)
    result_string = star_by_p_value(p_value, '{:.4f}±{:.4f}'.format(np.mean(other_model_values), np.std(other_model_values)))
    return result_string