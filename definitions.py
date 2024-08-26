"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path
import os
import argparse
import time
from tqdm import tqdm
import copy
import pickle
import numpy as np
import pandas as pd
import random
import torch
import graphviz
import xgboost
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost.sklearn import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
# from imblearn.over_sampling import SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier, RUSBoostClassifier
# importing DCS techniques from DESlib
from deslib.dcs.ola import OLA
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
# import DES techniques from DESlib
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, \
    precision_recall_curve, auc, average_precision_score
from src.omni.functions import load_pickle, save_pickle
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shap
import warnings
from matplotlib import rc

rc('font', family='Palatino Linotype')
warnings.filterwarnings('ignore')

# full features
dem_cols = ['age', 'gender', 'Weight_kg', 're_admission', 'elixhauser']  # 5
state_cols = [
    # Vital signs
    'SOFA', 'SIRS', 'GCS', 'HR', 'SysBP',  # 5
    'MeanBP', 'DiaBP', 'Shock_Index', 'RR', 'SpO2', 'Temp_C',  # 6
    # Lab values
    'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium', 'Calcium',  # 8
    'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'WBC_count',  # 7
    'Platelets_count', 'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',  # 7
    'Arterial_BE', 'HCO3', 'Arterial_lactate', 'PaO2_FiO2', 'Hb'  # 5
    ]
Ventilation_params = ['mechvent', 'FiO2_1']  # 2
Fluid_cols = ['input_total', 'input_4hourly', 'output_total',
              'output_4hourly', 'max_dose_vaso', 'cumulated_balance']  # 6
outcome_cols = ['died_in_hosp', 'died_within_48h_of_out_time']

fold_number = 10
random.seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = str(Path(__file__).resolve().parents[0])
DATA_DIR = ROOT_DIR + '/data/'
MODELS_DIR = ROOT_DIR + '/models/'
RESULT_DIR = ROOT_DIR + '/results/'
SOURCE_DIR = ROOT_DIR + '/src/'
FIG_DIR = ROOT_DIR + '/FigSupp/'

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR + '/processed_data/', exist_ok=True)
    os.makedirs(DATA_DIR + '/raw_data/', exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(SOURCE_DIR + '/utils/', exist_ok=True)
    os.makedirs(SOURCE_DIR + '/data/', exist_ok=True)
    os.makedirs(SOURCE_DIR + '/model/', exist_ok=True)
    os.makedirs(SOURCE_DIR + '/evaluate/', exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
