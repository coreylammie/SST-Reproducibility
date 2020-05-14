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
lrs_parameters = lrs_model.fit_tempurature(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, r_on=True)
hrs_parameters = hrs_model.fit_tempurature(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, r_on=False)

print('lrs: %s' % lrs_parameters)
print('hrs: %s' % hrs_parameters)

plt.figure(1)
plt.xlim(0, 600)
# plt.xlim(100, 600)
# plt.xlim(200, 450)
plt.ylim(1e3, 1e5)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS', color='r')
plt.scatter(np.linspace(1, 600), lrs_model.model_tempurature_dependance(np.linspace(100, 600), **lrs_parameters), color='m')
# plt.scatter(lrs.iloc[:, 0], lrs_model.model_tempurature_dependance(lrs.iloc[:, 0].values, **lrs_parameters), color='m')
plt.scatter(np.linspace(1, 600), hrs_model.model_tempurature_dependance(np.linspace(100, 600), **hrs_parameters), color='m')
# plt.scatter(hrs.iloc[:, 0], hrs_model.model_tempurature_dependance(hrs.iloc[:, 0].values, **hrs_parameters), color='m')

plt.yscale('log')
plt.xlabel('Tempurature (K)')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()
plt.show()

# plt.figure(1)
# plt.xlim(200, 450)
# plt.ylim(1e3, 1e5)
# plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS', color='b')
# plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS', color='r')
# plt.yscale('log')
# plt.xlabel('Tempurature (K)')
# plt.ylabel('Resistance ($\Omega$)')
# plt.grid(b=True, axis='both')
# plt.minorticks_on()
# plt.show()
