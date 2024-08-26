# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/8 22:22
@Auth ： Hongwei
@File ：model_utils.py
@IDE ：PyCharm
"""
from definitions import *


def compute_weight(distances, beta):
    distances = np.exp(distances * beta)
    weight = np.reciprocal(distances) / np.sum(np.reciprocal(distances), axis=1, keepdims=True)
    return weight