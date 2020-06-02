import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/3A_LRS_raw_data.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/3A_HRS_raw_data.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual)
    lrs_parameters = lrs_model.fit_tempurature(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, stable_resistance=2400, r_on=True)
    hrs_parameters = hrs_model.fit_tempurature(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, stable_resistance=55000, r_on=False)

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlim(200, 450)
plt.ylim(1e3, 1e5)
plt.yscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(np.linspace(150, 500), lrs_model.model_tempurature_dependence(np.linspace(150, 500), **lrs_parameters), label='LRS Model', linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(np.linspace(150, 500), hrs_model.model_tempurature_dependence(np.linspace(150, 500), **hrs_parameters), label='HRS Model', linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Tempurature (K)', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.legend()
plt.show()
