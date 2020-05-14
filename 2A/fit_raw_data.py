import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


lrs_10 = pd.read_csv('LRS_10_raw_data.csv')
lrs_20 = pd.read_csv('LRS_20_raw_data.csv')
lrs_30 = pd.read_csv('LRS_30_raw_data.csv')
hrs_10 = pd.read_csv('HRS_10_raw_data.csv')
hrs_20 = pd.read_csv('HRS_20_raw_data.csv')
hrs_30 = pd.read_csv('HRS_30_raw_data.csv')
lrs = [lrs_10, lrs_20, lrs_30]
hrs = [hrs_10, hrs_20, hrs_30]
cell_sizes = [10, 20, 30]

# Cell size dependence
lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
lrs_parameters = lrs_model.fit(raw_data_x=[lrs_10.iloc[:, 0].values, lrs_20.iloc[:, 0].values],
                           raw_data_y=[lrs_10.iloc[:, 1].values, lrs_20.iloc[:, 1].values],
                           stable_resistance=4400,
                           threshold=[1e4, 1e7],
                           cell_size=[10, 20])
hrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
hrs_parameters = hrs_model.fit(raw_data_x=[hrs_10.iloc[:, 0].values, hrs_20.iloc[:, 0].values],
                           raw_data_y=[hrs_10.iloc[:, 1].values, hrs_20.iloc[:, 1].values],
                           stable_resistance=65000,
                           threshold=[1e4, 1e7],
                           cell_size=[10, 20])

for i in range(len(cell_sizes)):
    plt.figure(i)
    plt.xlim(1e2, 1e9)
    plt.ylim(1e3, 1e6)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Cell size = %dnm' % cell_sizes[i])
    # Plot raw_data
    plt.scatter(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, label='LRS Experimental', color='b')
    plt.scatter(hrs[i].iloc[:, 0].values, hrs[i].iloc[:, 1].values, label='HRS Experimental', color='r')
    # Plot fitted_model
    plt.scatter(lrs[i].iloc[:, 0].values, lrs_model.model_gradual_convergence(lrs[i].iloc[:, 0].values, lrs_parameters['pq_0'], lrs_parameters['pq_1'], lrs_parameters['pq_2_1'], lrs_parameters['pq_3'], cell_sizes[i]), label='LRS Model', color='b', marker='s')
    plt.scatter(hrs[i].iloc[:, 0].values, hrs_model.model_gradual_convergence(hrs[i].iloc[:, 0].values, hrs_parameters['pq_0'], hrs_parameters['pq_1'], hrs_parameters['pq_2_1'], hrs_parameters['pq_3'], cell_sizes[i]), label='HRS Model', color='r', marker='s')

# Single cell size
lrs_single_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
lrs_single_parameters = lrs_single_model.fit(raw_data_x=lrs_10.iloc[:, 0].values,
                           raw_data_y=lrs_10.iloc[:, 1].values,
                           stable_resistance=4400,
                           threshold=1e4,
                           cell_size=10)
hrs_single_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
hrs_single_parameters = lrs_single_model.fit(raw_data_x=hrs_10.iloc[:, 0].values,
                           raw_data_y=hrs_10.iloc[:, 1].values,
                           stable_resistance=65000,
                           threshold=1e4,
                           cell_size=10)

plt.figure(i + 1)
plt.xlim(1e2, 1e9)
plt.ylim(1e3, 1e6)
plt.xscale('log')
plt.yscale('log')
plt.title('Cell size = %dnm' % cell_sizes[0])
# Plot raw_data
plt.scatter(lrs[0].iloc[:, 0].values, lrs[0].iloc[:, 1].values, label='LRS Experimental', color='b')
plt.scatter(hrs[0].iloc[:, 0].values, hrs[0].iloc[:, 1].values, label='HRS Experimental', color='r')
# Plot fitted_model
plt.scatter(lrs[0].iloc[:, 0].values, lrs_model.model_gradual_convergence(lrs[0].iloc[:, 0].values, lrs_single_parameters['pq_0'], lrs_single_parameters['pq_1'], lrs_single_parameters['pq_2_1'], lrs_single_parameters['pq_3'], cell_sizes[0]), label='LRS Model', color='b', marker='s')
plt.scatter(hrs[0].iloc[:, 0].values, hrs_model.model_gradual_convergence(hrs[0].iloc[:, 0].values, hrs_single_parameters['pq_0'], hrs_single_parameters['pq_1'], hrs_single_parameters['pq_2_1'], hrs_single_parameters['pq_3'], cell_sizes[0]), label='HRS Model', color='r', marker='s')
plt.show()
