import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/2D_LRS_raw_data.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/2D_HRS_raw_data.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=False)
    hrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=False)
    lrs_threshold = {}
    lrs_threshold[(10, None)] = 0.
    lrs_model_parameters = lrs_model.fit(initial_resistance=2.00e4, stable_resistance=2.00e4, threshold=lrs_threshold)
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 2e7
    hrs_model_parameters = hrs_model.fit(initial_resistance=10.75e4, stable_resistance=2.00e4, threshold=hrs_threshold)
    lrs_model_output = lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters)
    hrs_model_output = hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters)

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.title('TiN/ETML/HfO$_x$/TiN', fontsize=label_size)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlim(1e2, 1e8)
plt.ylim(1e4, 12e4)
plt.xscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(lrs.iloc[:, 0].values, lrs_model_output, linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model_output, linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Cycle Number', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.show()
