import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
sys.path.insert(0,'..')
from GeneralModel import GeneralModel
from GeneralModel import OperationMode

# Constant compliance current- 200uA
fit_raw_data = True
# Import and concatenate experimental data
lrs_200 = pd.read_csv('200.csv', header=None)
lrs_200 = lrs_200.sort_values(by=lrs_200.columns[0])
lrs_225 = pd.read_csv('225.csv', header=None)
lrs_225 = lrs_225.sort_values(by=lrs_225.columns[0])
lrs_250 = pd.read_csv('250.csv', header=None)
lrs_250 = lrs_250.sort_values(by=lrs_250.columns[0])
lrs = [lrs_200, lrs_225, lrs_250]
tempuratures = [200, 225, 250]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
# if fit_raw_data:
#     lrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=True)
#     lrs_model_parameters = lrs_model.fit(raw_data_x=[lrs_20.iloc[:, 0].values, lrs_30.iloc[:, 0].values, lrs_40.iloc[:, 0].values],
#                                raw_data_y=[lrs_20.iloc[:, 1].values, lrs_30.iloc[:, 1].values, lrs_40.iloc[:, 1].values],
#                                initial_resistance=300000,
#                                stable_resistance=5e7,
#                                threshold=[2e6, 1.5e7, 1.7e8],
#                                cell_size=cell_sizes)

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('TBD', fontsize=label_size)
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    # plt.xlim(1e0, 1e9)
    # plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    # if fit_raw_data:
    #     plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    #     plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
