import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


lrs = pd.read_csv('LRS_raw_data.csv')
hrs = pd.read_csv('HRS_raw_data.csv')

lrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=False)
hrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=False)
lrs_model_parameters = lrs_model.fit(initial_resistance=2.00e4, stable_resistance=2.00e4, threshold=0)
hrs_model_parameters = hrs_model.fit(initial_resistance=10.75e4, stable_resistance=2.00e4, threshold=2.00e7)
lrs_model_output = lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters)
hrs_model_output = hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters)

plt.figure(1)
plt.xlim(1e2, 1e8)
plt.ylim(1e4, 12e4)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', color='r')
plt.scatter(lrs.iloc[:, 0].values, lrs_model_output, label='LRS Model', color='b', marker='s')
plt.scatter(hrs.iloc[:, 0].values, hrs_model_output, label='HRS Model', color='r', marker='s')
plt.xscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()
plt.legend()
plt.show()
