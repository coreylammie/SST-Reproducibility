import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


# TODO extrapolate to all cell sizes
lrs = pd.read_csv('LRS_10_raw_data.csv')
hrs = pd.read_csv('HRS_10_raw_data.csv')

model = GeneralModel(operation_mode=OperationMode.gradual,
                     tempurature_dependance=False,
                     cell_size_dependance=False)

lrs_parameters = model.fit(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, 4400, 1e4)
hrs_parameters = model.fit(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, 50000, 1e5)
lrs_model = model.gradual_convergence(lrs.iloc[:, 0].values, **lrs_parameters)
hrs_model = model.gradual_convergence(hrs.iloc[:, 0].values, **hrs_parameters)

plt.figure(1)
plt.xlim(1e2, 1e9)
plt.ylim(1e3, 1e6)
# Raw Data
plt.scatter(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, label='LRS Experimental', color='b')
plt.scatter(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, label='HRS Experimental', color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Cycles')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
# Fitted Model
plt.scatter(lrs.iloc[:, 0].values, lrs_model, label='LRS Model', color='b', marker='s')
plt.scatter(hrs.iloc[:, 0].values, hrs_model, label='HRS Model', color='r', marker='s')
plt.minorticks_on()
plt.legend()
plt.show()
