import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
# import sys
# sys.path.insert(0,'..')
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs_200_O = pd.read_csv('Experimental Data/3B_LRS_O_200.csv', header=None)
lrs_200_O = lrs_200_O.sort_values(by=lrs_200_O.columns[0])
lrs_250_O = pd.read_csv('Experimental Data/3B_LRS_O_250.csv', header=None)
lrs_220_O = lrs_250_O.sort_values(by=lrs_250_O.columns[0])
lrs_300_O = pd.read_csv('Experimental Data/3B_LRS_O_300.csv', header=None)
lrs_300_O = lrs_300_O.sort_values(by=lrs_300_O.columns[0])
lrs_O = [lrs_200_O, lrs_250_O, lrs_300_O]
lrs_200_A = pd.read_csv('Experimental Data/3B_LRS_Al_200.csv', header=None)
lrs_200_A = lrs_200_A.sort_values(by=lrs_200_A.columns[0])
lrs_250_A = pd.read_csv('Experimental Data/3B_LRS_Al_250.csv', header=None)
lrs_220_A = lrs_250_A.sort_values(by=lrs_250_A.columns[0])
lrs_300_A = pd.read_csv('Experimental Data/3B_LRS_Al_300.csv', header=None)
lrs_300_A = lrs_300_A.sort_values(by=lrs_300_A.columns[0])
lrs_A = [lrs_200_A, lrs_250_A, lrs_300_A]
tempuratures = [273+200, 273+250, 273+300]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_O_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_O_model_parameters = {'initial_resistance': 4250, 'p_1': 14610000000, 'p_2': -1.9784220000000001, 'p_3': 0.14041884744983046, 'tempurature_threshold': 298}
    lrs_A_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_A_model_parameters = {'initial_resistance': 4250, 'p_1': 790100000000, 'p_2': -2.5029019999999997, 'p_3': 0.0577995548620432257, 'tempurature_threshold': 298}

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 20
tick_size = 16
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('Ti/HfO$_x$/TiN\nTi/HfAlO/TiN', fontsize=label_size)
markers = ['s', '^', 'v']
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    # plt.xlim(1e0, 1e9)
    # plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs_O[i].iloc[:, 0].values, lrs_O[i].iloc[:, 1].values, linestyle='-', color='b', marker=markers[i], markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(lrs_A[i].iloc[:, 0].values, lrs_A[i].iloc[:, 1].values, linestyle='-', color='magenta', marker=markers[i], markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs_O[i].iloc[:, 0].values, lrs_O_model.model(lrs_O[i].iloc[:, 0].values, **lrs_O_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(lrs_A[i].iloc[:, 0].values, lrs_A_model.model(lrs_A[i].iloc[:, 0].values, **lrs_A_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='magenta', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
    plt.gca().tick_params(axis='both', which='minor', labelsize=tick_size)

plt.show()
