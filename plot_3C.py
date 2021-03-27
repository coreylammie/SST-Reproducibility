import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/3C_LRS.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/3C_HRS.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, temperature_dependance=True, cell_size_dependance=False)
    lrs_raw_data_x = {}
    lrs_raw_data_x[(10, None)] = lrs.iloc[:, 0].values
    lrs_raw_data_y = {}
    lrs_raw_data_y[(10, None)] = lrs.iloc[:, 1].values
    lrs_threshold = {}
    lrs_threshold[(10, None)] = 1e3
    lrs_model_parameters = lrs_model.fit(raw_data_x=lrs_raw_data_x,
                                         raw_data_y=lrs_raw_data_y,
                                         initial_resistance=8.8e4,
                                         threshold=lrs_threshold)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual, temperature_dependance=True, cell_size_dependance=False)
    hrs_raw_data_x = {}
    hrs_raw_data_x[(10, None)] = hrs.iloc[:, 0].values
    hrs_raw_data_y = {}
    hrs_raw_data_y[(10, None)] = hrs.iloc[:, 1].values
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 1e3
    hrs_model_parameters = hrs_model.fit(raw_data_x=hrs_raw_data_x,
                                         raw_data_y=hrs_raw_data_y,
                                         initial_resistance=330000,
                                         threshold=hrs_threshold)

# Plot the experimental data and results from the model
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 20
tick_size = 16
plt.figure(1)
plt.title('TiN/HfO$_x$/TiN', fontsize=label_size)
plt.xlim(4e1, 2e6)
plt.ylim(5e4, 5e5)
plt.xscale('log')
plt.yscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Time (s)', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.gca().tick_params(axis='both', which='minor', labelsize=tick_size)
plt.axvspan(1200, 1e6, facecolor='b', alpha=0.25)
plt.axvline(x=1200)
plt.axvline(x=1e6)
plt.show()
