# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/10 9:07
@Auth ： Hongwei
@File ：2_split_sepsis_cohort.py
@IDE ：PyCharm
"""
from definitions import *


def split_sepsis_cohort(reward_label):
    if reward_label == 'died_in_hosp':
        SAVE_DIR = DATA_DIR + '/processed_data/hospital_mortality/'
    elif reward_label == 'died_within_48h_of_out_time':
        SAVE_DIR = DATA_DIR + '/processed_data/48h_mortality/'
    elif reward_label == 'mortality_90d':
        SAVE_DIR = DATA_DIR + '/processed_data/90d_mortality/'

    print('\n----------------------------{}----------------------------\n'.format(reward_label))
    folds = 10
    fullzs_data_path = os.path.join(SAVE_DIR, 'MIMIC_zs.csv')
    full_zs = pd.read_csv(fullzs_data_path)
    full_zs = full_zs.drop(['presumed_onset', 'charttime', 'icustayid'], axis=1)

    old_cols = ['gender', 'age', 'Weight_kg', 're_admission', 'elixhauser',
                'SOFA', 'SIRS', 'GCS', 'HR', 'SysBP', 'MeanBP',
                'DiaBP', 'Shock_Index', 'RR', 'SpO2', 'Temp_C',
                'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium',
                'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili',
                'Albumin', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'INR',
                'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3',
                'Arterial_lactate', 'PaO2_FiO2', 'Hb',
                'mechvent', 'FiO2_1',
                'input_total', 'input_4hourly', 'output_total', 'output_4hourly',
                'max_dose_vaso', 'cumulated_balance'] + [reward_label]
    full_zs = full_zs[old_cols]
    alive_patient, died_patient = full_zs[full_zs.iloc[:, -1] == 0], full_zs[full_zs.iloc[:, -1] == 1]
    print('Number of Positive SampleL {}'.format(died_patient.shape[0]))
    died_patient = died_patient.sample(n=19000, random_state=0).reset_index(drop=True)

    new_cols = ['Gender', 'Age', 'Weight', 'Readmission', 'Elixhauser',
                'SOFA', 'SIRS', 'GCS', 'HR', 'SystolicBP', 'MeanBP',
                'DiastolicBP', 'Shock Index', 'RR', 'SpO2', 'Temperature',
                'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium',
                'Calcium', 'IonizedCa', 'CO2', 'SGOT', 'SGPT', 'TotalBilirubin',
                'Albumin', 'WBC', 'Platelets', 'PTT', 'PT', 'INR',
                'ArterialPH', 'PaO2', 'PaCO2', 'ArterialBE', 'HCO3',
                'ArterialLactate', 'PaO2/FiO2', 'Hb',
                'Mechanical ventilation', 'FiO2',
                'Total input', '4 hourly input', 'Total output', '4 hourly output',
                'Vasopressor', 'CumulatedBalance', 'mortality_label']

    for ratio in [1, 3, 5]:
        print('------------------ratio 1:{}------------------'.format(ratio))
        positive_patient = alive_patient.sample(n=died_patient.shape[0] * ratio)
        negative_patient = died_patient
        balanced_data = pd.concat([positive_patient, negative_patient], axis=0).reset_index(drop=True)

        y = balanced_data.values[:, -1]
        X = balanced_data.index.values

        cur_ratio_dataset = {}
        for fold_idx in range(folds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                                test_size=0.4, random_state=fold_idx)
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test,
                                                            test_size=0.5, random_state=fold_idx)

            train_data = balanced_data[balanced_data.index.isin(X_train)]
            val_data = balanced_data[balanced_data.index.isin(X_val)]
            test_data = balanced_data[balanced_data.index.isin(X_test)]

            fold_data = {'cols': new_cols,
                         'train': train_data.values,
                         'val': val_data.values,
                         'test': test_data.values}
            cur_ratio_dataset['fold {}'.format(fold_idx + 1)] = fold_data

        save_pickle(cur_ratio_dataset, SAVE_DIR + '{}_folds_datasets_BR-[1-{}].pickle'.format(folds, ratio))


if __name__ == '__main__':
    rewards = ['died_in_hosp', 'died_within_48h_of_out_time']
    # 27626 19027
    for reward_label in rewards:
        split_sepsis_cohort(reward_label=reward_label)
