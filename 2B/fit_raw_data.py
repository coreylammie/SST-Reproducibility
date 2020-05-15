import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


hrs = pd.read_csv('HRS_raw_data.csv')
lrs = pd.read_csv('LRS_raw_data.csv')

plt.figure(1)
plt.xlim(1e2, 1e8)
plt.ylim(1e4, 12e4)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', color='r')
plt.xscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()

hrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=True)
plt.scatter(hrs.iloc[:, 0].values, hrs_model.model_sudden_convergence(hrs.iloc[:, 0].values, np.log10(10.75e4), np.log10(2e4), threshold=2e7), color='m')

plt.show()
