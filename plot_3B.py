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
lrs_200_O = pd.read_csv('Experimental Data/3B_LRS_O_200_raw_data.csv', header=None)
lrs_200_O = lrs_200_O.sort_values(by=lrs_200_O.columns[0])
lrs_250_O = pd.read_csv('Experimental Data/3B_LRS_O_250_raw_data.csv', header=None)
lrs_220_O = lrs_250_O.sort_values(by=lrs_250_O.columns[0])
lrs_300_O = pd.read_csv('Experimental Data/3B_LRS_O_300_raw_data.csv', header=None)
lrs_300_O = lrs_300_O.sort_values(by=lrs_300_O.columns[0])
lrs_O = [lrs_200_O, lrs_250_O, lrs_300_O]
lrs_200_A = pd.read_csv('Experimental Data/3B_LRS_Al_200_raw_data.csv', header=None)
lrs_200_A = lrs_200_A.sort_values(by=lrs_200_A.columns[0])
lrs_250_A = pd.read_csv('Experimental Data/3B_LRS_Al_250_raw_data.csv', header=None)
lrs_220_A = lrs_250_A.sort_values(by=lrs_250_A.columns[0])
lrs_300_A = pd.read_csv('Experimental Data/3B_LRS_Al_300_raw_data.csv', header=None)
lrs_300_A = lrs_300_A.sort_values(by=lrs_300_A.columns[0])
lrs_A = [lrs_200_A, lrs_250_A, lrs_300_A]
tempuratures = [273+200, 273+250, 273+300]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_O_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_O_model_parameters = {'initial_resistance': 4250, 'p_1': 5.471e19, 'p_2': -2.195664, 'p_3': 0.11622935929992417, 'tempurature_threshold': 298}
    lrs_A_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_A_model_parameters = {'initial_resistance': 4250, 'p_1': 2.633e19, 'p_2': -2.19328, 'p_3': 0.07677623515706467, 'tempurature_threshold': 298}

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('Ti/HfO$_x$/TiN\nTi/HfAlO/TiN', fontsize=label_size)
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    # plt.xlim(1e0, 1e9)
    # plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs_O[i].iloc[:, 0].values, lrs_O[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(lrs_A[i].iloc[:, 0].values, lrs_A[i].iloc[:, 1].values, linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs_O[i].iloc[:, 0].values, lrs_O_model.model(lrs_O[i].iloc[:, 0].values, **lrs_O_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(lrs_A[i].iloc[:, 0].values, lrs_A_model.model(lrs_A[i].iloc[:, 0].values, **lrs_A_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
