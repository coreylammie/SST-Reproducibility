import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


hrs = pd.read_csv('HRS_raw_data.csv')
lrs = pd.read_csv('LRS_raw_data.csv')

lrs_model = GeneralModel(operation_mode=OperationMode.gradual)
hrs_model = GeneralModel(operation_mode=OperationMode.gradual)
lrs_parameters = lrs_model.fit_tempurature(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, stable_resistance=2400, r_on=True)
hrs_parameters = hrs_model.fit_tempurature(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, stable_resistance=55000, r_on=False)

plt.figure(1)
plt.xlim(150, 500)
plt.ylim(1e3, 1e5)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', color='r')
plt.scatter(np.linspace(150, 500), lrs_model.model_tempurature_dependence(np.linspace(150, 500), **lrs_parameters), label='LRS Model', color='b', marker='s')
plt.scatter(np.linspace(150, 500), hrs_model.model_tempurature_dependence(np.linspace(150, 500), **hrs_parameters), label='HRS Model', color='r', marker='s')
plt.yscale('log')
plt.xlabel('Tempurature (K)')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()
plt.legend()
plt.show()
