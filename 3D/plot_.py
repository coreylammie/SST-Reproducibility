import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
sys.path.insert(0,'..')
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs_200 = pd.read_csv('A_200.csv', header=None)
lrs_200 = lrs_200.sort_values(by=lrs_200.columns[0])
lrs_250 = pd.read_csv('A_250.csv', header=None)
lrs_220 = lrs_250.sort_values(by=lrs_250.columns[0])
lrs_300 = pd.read_csv('A_300.csv', header=None)
lrs_300 = lrs_300.sort_values(by=lrs_300.columns[0])
lrs = [lrs_200, lrs_250, lrs_300]
tempuratures = [273+200, 273+250, 273+300]


# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    # lrs_raw_data_x = {}
    # lrs_raw_data_x[(10, 273+200)] = lrs_200.iloc[:, 0].values
    # lrs_raw_data_x[(10, 273+250)] = lrs_250.iloc[:, 0].values
    # lrs_raw_data_x[(10, 273+300)] = lrs_300.iloc[:, 0].values
    # lrs_raw_data_y = {}
    # lrs_raw_data_y[(10, 273+200)] = lrs_200.iloc[:, 1].values
    # lrs_raw_data_y[(10, 273+250)] = lrs_250.iloc[:, 1].values
    # lrs_raw_data_y[(10, 273+300)] = lrs_300.iloc[:, 1].values
    # lrs_threshold = {}
    # lrs_threshold[(10, 273+200)] = 30000
    # lrs_threshold[(10, 273+250)] = 500
    # lrs_threshold[(10, 273+300)] = 100
    # lrs_model_parameters = lrs_model.fit(raw_data_x=lrs_raw_data_x,
    #                        raw_data_y=lrs_raw_data_y,
    #                        initial_resistance=4250,
    #                        threshold=lrs_threshold,
    #                        tempurature=tempuratures,
    #                        tempurature_threshold=298)
    lrs_model_parameters = {'initial_resistance': 4250, 'p_1': 2.633e19, 'p_2': -2.19328, 'p_3': 0.07677623515706467, 'tempurature_threshold': 298}

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('Ti/HfAlO/TiN', fontsize=label_size)
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    # plt.xlim(1e0, 1e9)
    # plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        x = np.logspace(0, 6, 50)
        # plt.plot(x, lrs_model.model(x, **lrs_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
