import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/2B_LRS.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/2B_HRS.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=False)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=False)
    hrs_data_x = {}
    hrs_data_x[(10, None)] = hrs.iloc[:, 0].values
    hrs_data_y = {}
    hrs_data_y[(10, None)] = hrs.iloc[:, 1].values
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 120
    hrs_model_parameters = hrs_model.fit(raw_data_x=hrs_data_x,
                                raw_data_y=hrs_data_y,
                                initial_resistance=9e10,
                                threshold=hrs_threshold)
    lrs_model_parameters = {'initial_resistance': 750, 'p_0': 0.5, 'p_1': 0., 'p_2': 0.}

# Plot the experimental data and results from the model
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 20
tick_size = 16
plt.figure(1)
plt.title('Cu/HfO$_x$/Pt', fontsize=label_size)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlim(1e1, 1e8)
plt.ylim(1e2, 1e12)
plt.xscale('log')
plt.yscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(lrs.iloc[:, 0].values, lrs_model.model(hrs.iloc[:, 0].values, **lrs_model_parameters, cell_size=None), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters, cell_size=None), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Cycle Number', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.gca().tick_params(axis='both', which='minor', labelsize=tick_size)
plt.show()
