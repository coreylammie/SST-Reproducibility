import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
# import sys
# sys.path.insert(0,'..')
from GeneralModel import GeneralModel
from GeneralModel import OperationMode

# Constant compliance current- 200uA
fit_raw_data = True
# Import and concatenate experimental data
lrs_200 = pd.read_csv('Experimental Data/3A_LRS_200.csv', header=None)
lrs_200 = lrs_200.sort_values(by=lrs_200.columns[0])
lrs_225 = pd.read_csv('Experimental Data/3A_LRS_225.csv', header=None)
lrs_225 = lrs_225.sort_values(by=lrs_225.columns[0])
lrs_250 = pd.read_csv('Experimental Data/3A_LRS_250.csv', header=None)
lrs_250 = lrs_250.sort_values(by=lrs_250.columns[0])
lrs = [lrs_250, lrs_225, lrs_200]
tempuratures = [200+273, 225+273, 250+273]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.sudden)
    initial_resistance = 40000
    stable_resistance = 2.6e7
    tempurature_threshold = 298
    lrs_model_parameters = {'initial_resistance': initial_resistance, 'p_1': np.log10(stable_resistance), 'p_2': 6.849e-10, 'p_3': 1.675, 'tempurature_threshold': tempurature_threshold}

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('Pt/Cu:MoO$_x$/GdO$_x$/Pt', fontsize=label_size)
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    plt.xlim(1e0, 1e4)
    plt.ylim(1e4, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, tempurature=tempuratures[i], **lrs_model_parameters), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
