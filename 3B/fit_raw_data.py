import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


lrs = pd.read_csv('LRS_raw_data.csv')
hrs = pd.read_csv('HRS_raw_data.csv')

lrs_model = GeneralModel(operation_mode=OperationMode.gradual, tempurature_dependance=True, cell_size_dependance=False)
# hrs_segment_model = GeneralModel(operation_mode=OperationMode.gradual, tempurature_dependance=True, cell_size_dependance=False)
lrs_model_parameters = lrs_model.fit(raw_data_x=lrs.iloc[:, 0].values,
                                     raw_data_y=lrs.iloc[:, 1].values,
                                     stable_resistance=9e4,
                                     threshold=1e3)

print(lrs_model_parameters)

plt.figure(1)
plt.xlim(1e0, 1e7)
plt.ylim(1e4, 1e6)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', color='r')
plt.scatter(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), label='LRS Model', color='b', marker='s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()
plt.show()






#
# plt.figure(1)
# plt.xlim(1e0, 1e7)
# plt.ylim(1e4, 1e6)
# plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS', color='b')
# plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS', color='r')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Time (s)')
# plt.ylabel('Resistance ($\Omega$)')
# plt.grid(b=True, axis='both')
# plt.minorticks_on()
# plt.show()
