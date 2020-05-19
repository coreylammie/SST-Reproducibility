import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


lrs = pd.read_csv('LRS_raw_data.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('HRS_raw_data.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

lrs_model = GeneralModel(operation_mode=OperationMode.gradual)
hrs_model = GeneralModel(operation_mode=OperationMode.gradual)
lrs_parameters = lrs_model.fit_tempurature(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, stable_resistance=2400, r_on=True)
hrs_parameters = hrs_model.fit_tempurature(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, stable_resistance=55000, r_on=False)

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
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(np.linspace(150, 500), lrs_model.model_tempurature_dependence(np.linspace(150, 500), **lrs_parameters), label='LRS Model', linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
plt.plot(np.linspace(150, 500), hrs_model.model_tempurature_dependence(np.linspace(150, 500), **hrs_parameters), label='HRS Model', linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
plt.yscale('log')
plt.xlabel('Tempurature (K)', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.show()
