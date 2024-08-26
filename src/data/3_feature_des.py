# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/30 11:06
@Auth ： Hongwei
@File ：3_feature_des.py
@IDE ：PyCharm
"""
from definitions import *

if __name__ == '__main__':
    df = pd.read_csv(r'D:\Research\Database\healthcare/MIMICtable_version_1.csv')
    gps = df.groupby('icustayid')
    genders, ages, weights, re_admissions, elixhausers, died_48h, died_in_hosp = [], [], [], [], [], [], []
    for idx, gp in gps:
        genders.append(gp['gender'].mean())
        ages.append(gp['age'].mean() / 365)
        weights.append(gp['Weight_kg'].mean())
        re_admissions.append(gp['re_admission'].mean())
        elixhausers.append(gp['elixhauser'].mean())
        died_48h.append(gp['died_within_48h_of_out_time'].mean())
        died_in_hosp.append(gp['died_in_hosp'].mean())

    patient_numer = len(gps)
    Dems = np.array([['All patients', 'All states', 'Male (N,%)', 'Age', 'Weight', 'Readmission (N,%)', 'Elixhauser'],
                     [patient_numer,
                      df.shape[0],
                      '{} ({:.2f}%)'.format(np.count_nonzero(np.array(genders) == 0),
                                            np.count_nonzero(np.array(genders) == 0) / patient_numer * 100),
                      '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(ages), np.quantile(ages, 0.25),
                                                       np.quantile(ages, 0.75)),
                      '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(weights), np.quantile(weights, 0.25),
                                                       np.quantile(weights, 0.75)),
                      '{} ({:.2f}%)'.format(np.count_nonzero(np.array(re_admissions) == 1),
                                            np.count_nonzero(np.array(re_admissions) == 1) / patient_numer * 100),
                      '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(elixhausers), np.quantile(elixhausers, 0.25),
                                                       np.quantile(elixhausers, 0.75))]]).T

    state_list = []
    for col in state_cols:
        state_list.append(
            '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(df[col]), np.quantile(df[col], 0.25), np.quantile(df[col], 0.75)))
    States = np.array([state_cols, state_list]).T

    Mechvent = np.array(
        [['Mechanical ventilation (N,%)', 'FiO2'],
         ['{} ({:.2f}%)'.format(np.count_nonzero(np.array(df['mechvent']) == 1),
                                np.count_nonzero(np.array(df['mechvent']) == 1) / df.shape[0] * 100),
          '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(df['FiO2_1']), np.quantile(df['FiO2_1'], 0.25), np.quantile(df['FiO2_1'], 0.75))]]).T
    fluid_list = []
    for col in Fluid_cols:
        if col == 'max_dose_vaso':
            series = df[col]
            fluid_list.append(
                '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(series), np.quantile(series, 0.25), np.quantile(series, 0.75)))
        else:
            fluid_list.append(
                '{:.2f} ({:.2f}, {:.2f})'.format(np.mean(df[col]), np.quantile(df[col], 0.25), np.quantile(df[col], 0.75)))
    Fluids = np.array([Fluid_cols, fluid_list]).T

    Outcome = np.array(
        [['48 hours mortality (N,%)', 'In-hospital mortality (N,%)'],
         ['{} ({:.2f}%)'.format(np.count_nonzero(np.array(died_48h) == 1),
                                np.count_nonzero(np.array(died_48h) == 1) / patient_numer * 100),
          '{} ({:.2f}%)'.format(np.count_nonzero(np.array(died_in_hosp) == 1),
                                np.count_nonzero(np.array(died_in_hosp) == 1) / patient_numer * 100)]]).T
    Feature_des = np.vstack((Dems, States, Mechvent, Fluids, Outcome))
    Feature_des[:, 0] = ['All patients', 'All states', 'Male (n,%)', 'Age', 'Weight', 'Readmission (n,%)', 'Elixhauser',

                         'SOFA', 'SIRS', 'GCS', 'HR', 'SystolicBP', 'MeanBP',
                         'DiastolicBP', 'Shock Index', 'RR', 'SpO2', 'Temperature',

                         'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium',
                         'Calcium', 'IonizedCa', 'CO2', 'SGOT', 'SGPT', 'TotalBilirubin',
                         'Albumin', 'WBC', 'Platelets', 'PTT', 'PT', 'INR',
                         'ArterialPH', 'PaO2', 'PaCO2', 'ArterialBE', 'HCO3',
                         'ArterialLactate', 'PaO2/FiO2', 'Hb',

                         'Mechanical ventilation (n,%)', 'FiO2',

                         'Total input', '4 hourly input', 'Total output', '4 hourly output',
                         'Vasopressor', 'CumulatedBalance',

                         '48 hours mortality (n,%)', 'In-hospital mortality (n,%)']

    save_df = pd.DataFrame(Feature_des, columns=['Feature', 'Mean(Q1, Q3)'])
    save_df.to_excel(TABLE_DIR + 'Feature_des.xlsx', index=False)
    pass
