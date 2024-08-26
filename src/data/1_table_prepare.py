# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/10 9:07
@Auth ： Hongwei
@File ：1_table_prepare.py
@IDE ：PyCharm
"""
from definitions import *


def Table_prepare(reward_label):
    if reward_label == 'r:reward hospital mortality':
        save_dir = DATA_DIR + '/processed_data/hospital_mortality/'
        outcome_key = ['died_in_hosp']
    elif reward_label == 'r:reward 48h mortality':
        save_dir = DATA_DIR + '/processed_data/48h_mortality/'
        outcome_key = ['died_within_48h_of_out_time']
    elif reward_label == 'r:reward 90d mortality':
        save_dir = DATA_DIR + '/processed_data/90d_mortality/'
        outcome_key = ['mortality_90d']
    os.makedirs(save_dir, exist_ok=True)

    MIMICtable = pd.read_csv(r'D:\Research\Database\healthcare/MIMICtable_version_1.csv')

    # all 51 columns of interest
    colmeta = ['presumed_onset', 'charttime', 'icustayid']  # Meta-data around patient stay  3
    colbin = ['gender', 're_admission', 'mechvent']  # Binary features  3
    # Patient features that will be z-normalize  36
    colnorm = ['age', 'elixhauser', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1',
               'Potassium', 'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL',
               'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2',
               'Arterial_BE', 'HCO3', 'Arterial_lactate', 'SOFA', 'SIRS', 'Shock_Index',
               'PaO2_FiO2', 'cumulated_balance']
    # Patient features that will be log-normalized  12
    collog = ['SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',
              'max_dose_vaso', 'input_4hourly', 'input_total', 'output_total', 'output_4hourly']

    MIMICraw = MIMICtable[colmeta + colbin + colnorm + collog + outcome_key].values  # RAW values

    MIMICzs = np.hstack(
        [MIMICtable[colmeta].values, MIMICtable[colbin].values - 0.5, stats.zscore(MIMICtable[colnorm].values),
         stats.zscore(np.log(0.0001 + MIMICtable[collog].values)), MIMICtable[outcome_key].values.reshape(-1, 1)]
    )

    # Processed table
    MIMIC_zs = pd.DataFrame(MIMICzs, columns=colmeta + colbin + colnorm + collog + outcome_key)
    MIMIC_zs.to_csv(save_dir + 'MIMIC_zs.csv', index=False)

    # Raw table
    MIMIC_raw = pd.DataFrame(MIMICraw, columns=colmeta + colbin + colnorm + collog + outcome_key)
    MIMIC_raw.to_csv(save_dir + 'MIMIC_raw.csv', index=False)


if __name__ == '__main__':
    rewards = ['r:reward hospital mortality', 'r:reward 48h mortality']
    for reward_label in rewards:
        Table_prepare(reward_label=reward_label)
