import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs_10 = pd.read_csv('Experimental Data/2A_LRS_10.csv')
lrs_10 = lrs_10.sort_values(by=lrs_10.columns[0])
lrs_20 = pd.read_csv('Experimental Data/2A_LRS_20.csv')
lrs_20 = lrs_20.sort_values(by=lrs_20.columns[0])
hrs_10 = pd.read_csv('Experimental Data/2A_HRS_10.csv')
hrs_10 = hrs_10.sort_values(by=hrs_10.columns[0])
hrs_20 = pd.read_csv('Experimental Data/2A_HRS_20.csv')
hrs_20 = hrs_20.sort_values(by=hrs_20.columns[0])
lrs = [lrs_10, lrs_20]
hrs = [hrs_10, hrs_20]
cell_sizes = [10, 20]

# Fit the model in gradual operation mode to the 10nm and 20nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_raw_data_x = {}
    lrs_raw_data_x[(10, None)] = lrs_10.iloc[:, 0].values
    lrs_raw_data_x[(20, None)] = lrs_20.iloc[:, 0].values
    lrs_raw_data_y = {}
    lrs_raw_data_y[(10, None)] = lrs_10.iloc[:, 1].values
    lrs_raw_data_y[(20, None)] = lrs_20.iloc[:, 1].values
    lrs_threshold = {}
    lrs_threshold[(10, None)] = 1e4
    lrs_threshold[(20, None)] = 1e7
    lrs_model_parameters = lrs_model.fit(raw_data_x=lrs_raw_data_x,
                               raw_data_y=lrs_raw_data_y,
                               initial_resistance=4400,
                               threshold=lrs_threshold,
                               cell_size=cell_sizes)
    hrs_raw_data_x = {}
    hrs_raw_data_x[(10, None)] = hrs_10.iloc[:, 0].values
    hrs_raw_data_x[(20, None)] = hrs_20.iloc[:, 0].values
    hrs_raw_data_y = {}
    hrs_raw_data_y[(10, None)] = hrs_10.iloc[:, 1].values
    hrs_raw_data_y[(20, None)] = hrs_20.iloc[:, 1].values
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 1e4
    hrs_threshold[(20, None)] = 1e7
    hrs_model_parameters = hrs_model.fit(raw_data_x=hrs_raw_data_x,
                               raw_data_y=hrs_raw_data_y,
                               initial_resistance=65000,
                               threshold=hrs_threshold,
                               cell_size=cell_sizes)

    print(lrs_model_parameters)
    print(hrs_model_parameters)
    
# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 20
tick_size = 16
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('TiN/Hf(Al)O/Hf/TiN', fontsize=label_size)
markers = ['s', '^', 'v']
for i in range(len(cell_sizes)):
    plt.grid(b=True, which='both')
    plt.xlim(1e2, 1e9)
    plt.ylim(2e3, 1.5e5)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker=markers[i], markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(hrs[i].iloc[:, 0].values, hrs[i].iloc[:, 1].values, linestyle='-', color='r', marker=markers[i], markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(hrs[i].iloc[:, 0].values, hrs_model.model(hrs[i].iloc[:, 0].values, **hrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Cycle Number', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
    plt.gca().tick_params(axis='both', which='minor', labelsize=tick_size)

plt.show()
