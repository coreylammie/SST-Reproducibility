import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs_20 = pd.read_csv('Experimental Data/2C_LRS_20_raw_data.csv', header=None)
lrs_20 = lrs_20.sort_values(by=lrs_20.columns[0])
lrs_30 = pd.read_csv('Experimental Data/2C_LRS_30_raw_data.csv', header=None)
lrs_30 = lrs_30.sort_values(by=lrs_30.columns[0])
lrs_40 = pd.read_csv('Experimental Data/2C_LRS_40_raw_data.csv', header=None)
lrs_40 = lrs_40.sort_values(by=lrs_40.columns[0])
hrs_20 = pd.read_csv('Experimental Data/2C_HRS_20_raw_data.csv', header=None)
hrs_20 = hrs_20.sort_values(by=hrs_20.columns[0])
hrs_30 = pd.read_csv('Experimental Data/2C_HRS_30_raw_data.csv', header=None)
hrs_30 = hrs_30.sort_values(by=hrs_30.columns[0])
hrs_40 = pd.read_csv('Experimental Data/2C_HRS_40_raw_data.csv', header=None)
hrs_40 = hrs_40.sort_values(by=hrs_40.columns[0])
lrs = [lrs_20, lrs_30, lrs_40]
hrs = [hrs_20, hrs_30, hrs_40]
cell_sizes = [20, 30, 40]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=True)
    hrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=True)
    lrs_model_parameters = lrs_model.fit(raw_data_x=[lrs_20.iloc[:, 0].values, lrs_30.iloc[:, 0].values, lrs_40.iloc[:, 0].values],
                               raw_data_y=[lrs_20.iloc[:, 1].values, lrs_30.iloc[:, 1].values, lrs_40.iloc[:, 1].values],
                               initial_resistance=14000,
                               stable_resistance=5e7,
                               threshold=[2.4e6, 2e7, 2.5e8],
                               cell_size=cell_sizes)
    hrs_model_parameters = hrs_model.fit(raw_data_x=[hrs_20.iloc[:, 0].values, hrs_30.iloc[:, 0].values, hrs_40.iloc[:, 0].values],
                               raw_data_y=[hrs_20.iloc[:, 1].values, hrs_30.iloc[:, 1].values, hrs_40.iloc[:, 1].values],
                               initial_resistance=300000,
                               stable_resistance=5e7,
                               threshold=[2e6, 1.5e7, 1.7e8],
                               cell_size=cell_sizes)

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('TiN/Hf(Al)O/Hf/TiN', fontsize=label_size)
for i in range(len(cell_sizes)):
    plt.grid(b=True, which='both')
    plt.xlim(1e0, 1e9)
    plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(hrs[i].iloc[:, 0].values, hrs[i].iloc[:, 1].values, linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(hrs[i].iloc[:, 0].values, hrs_model.model(hrs[i].iloc[:, 0].values, **hrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Cycle Number', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
