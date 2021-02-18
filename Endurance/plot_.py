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
lrs_10 = pd.read_csv('10_L.csv', header=None)
lrs_20 = pd.read_csv('20_L.csv', header=None)
lrs_30 = pd.read_csv('30_L.csv', header=None)
lrs_40 = pd.read_csv('40_L.csv', header=None)
hrs_10 = pd.read_csv('10_R.csv', header=None)
hrs_20 = pd.read_csv('20_R.csv', header=None)
hrs_30 = pd.read_csv('30_R.csv', header=None)
hrs_40 = pd.read_csv('40_R.csv', header=None)
lrs_10 = lrs_10.sort_values(by=lrs_10.columns[0])
lrs_20 = lrs_20.sort_values(by=lrs_20.columns[0])
lrs_30 = lrs_30.sort_values(by=lrs_30.columns[0])
lrs_40 = lrs_40.sort_values(by=lrs_40.columns[0])
hrs_10 = hrs_10.sort_values(by=hrs_10.columns[0])
hrs_20 = hrs_20.sort_values(by=hrs_20.columns[0])
hrs_30 = hrs_30.sort_values(by=hrs_30.columns[0])
hrs_40 = hrs_40.sort_values(by=hrs_40.columns[0])
lrs = [lrs_10, lrs_20, lrs_30, lrs_40]
hrs = [hrs_10, hrs_20, hrs_30, hrs_40]
cell_sizes = [10, 20, 30, 40]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
# if fit_raw_data:
#     lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
#     hrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
#     lrs_model_parameters = lrs_model.fit(raw_data_x=[lrs_10.iloc[:, 0].values, lrs_20.iloc[:, 0].values],
#                                raw_data_y=[lrs_10.iloc[:, 1].values, lrs_20.iloc[:, 1].values],
#                                stable_resistance=4400,
#                                threshold=[1e4, 1e7],
#                                cell_size=[10, 20])
#     hrs_model_parameters = hrs_model.fit(raw_data_x=[hrs_10.iloc[:, 0].values, hrs_20.iloc[:, 0].values],
#                                raw_data_y=[hrs_10.iloc[:, 1].values, hrs_20.iloc[:, 1].values],
#                                stable_resistance=65000,
#                                threshold=[1e4, 1e7],
#                                cell_size=[10, 20])

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
    plt.xlim(1e1, 1e9)
    plt.ylim(1e5, 2e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(hrs[i].iloc[:, 0].values, hrs[i].iloc[:, 1].values, linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    # if fit_raw_data:
    #     plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    #     plt.plot(hrs[i].iloc[:, 0].values, hrs_model.model(hrs[i].iloc[:, 0].values, **hrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Cycle Number', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
