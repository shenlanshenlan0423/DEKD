# -*- coding: utf-8 -*-
"""
@Time ： 2024/1/21 9:22
@Auth ： Hongwei
@File ：cls_eval.py
@IDE ：PyCharm
"""
from definitions import *


def model_evaluate(model, X_test, y_test):
    pred_y_proba = model.predict_proba(X_test)[:, 1]

    pred_y = (pred_y_proba > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_y).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    res_dict = {
        'Accuracy': accuracy_score(y_test, pred_y),
        'Precision': precision_score(y_test, pred_y),
        'Recall': recall_score(y_test, pred_y),
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'FNR': fnr,
        'FPR': fpr,
        'F1': f1_score(y_test, pred_y, zero_division=0),
        'Balanced Accuracy': balanced_accuracy_score(y_test, pred_y),  # 对各类别的recall的平均值, 适用于正负样本比例严重失衡的情况
        'AUROC': roc_auc_score(y_test, pred_y_proba),
        'AUPRC': average_precision_score(y_test, pred_y_proba)
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
    result_string = star_by_p_value(p_value, '{:.4f}±{:.4f}'.format(np.mean(other_model_values), np.std(other_model_values)))  # Note: For Appendix text
    # result_string = star_by_p_value(p_value, '{:.4f}'.format(np.mean(other_model_values)))  # Note: For main text
    return result_string