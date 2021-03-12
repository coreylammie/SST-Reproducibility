import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# Fit the model in gradual operation mode
if fit_raw_data:
    lrs_O_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=False)
    lrs_O_model_parameters = {'initial_resistance': 4250, 'p_0': 1.541e-13, 'p_1': 0, 'p_2': 63.43, 'p_3': 0.04, 'tempurature_threshold': 298}
    lrs_A_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=False)
    lrs_A_model_parameters = {'initial_resistance': 4250, 'p_0': 4.764e-17, 'p_1': 0, 'p_2': 76.47, 'p_3': 0.014, 'tempurature_threshold': 298}

# Plot the experimental data and results from the model
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
