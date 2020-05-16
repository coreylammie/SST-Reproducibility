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
hrs_model = GeneralModel(operation_mode=OperationMode.gradual, tempurature_dependance=True, cell_size_dependance=False)
lrs_model_parameters = lrs_model.fit(raw_data_x=lrs.iloc[:, 0].values,
                                     raw_data_y=lrs.iloc[:, 1].values,
                                     stable_resistance=9e4,
                                     threshold=1e3)
hrs_model_parameters = hrs_model.fit(raw_data_x=hrs.iloc[:, 0].values,
                                     raw_data_y=hrs.iloc[:, 1].values,
                                     stable_resistance=330000,
                                     threshold=1e3)

h = plt.figure(1)
plt.xlim(1e1, 1e7)
plt.ylim(1e4, 1e6)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS Experimental', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS Experimental', color='r')
plt.scatter(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), label='LRS Model T=125$^{\circ}$C', color='b', marker='s')
plt.scatter(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), label='HRS Model T=125$^{\circ}$C', color='r', marker='s')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('Resistance ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()

# Fitting parameters from Fig. 3A: {'stable_resistance': 2400, 'p_0': 0.000801158151673717, 'stable_tempurature': 273}, {'stable_resistance': 55000, 'p_0': -0.00420717061486765, 'stable_tempurature': 273}
plt.figure(2)
plt.grid()
plt.xlim(150, 500)
plt.yscale('log')
general_model = GeneralModel(operation_mode=None)
plt.scatter(np.linspace(150, 500), general_model.model_tempurature_dependence(tempurature=np.linspace(150, 500), stable_resistance=9e4, p_0=0.000801158151673717 * (2400/9e4), stable_tempurature=298))
plt.scatter(np.linspace(150, 500), general_model.model_tempurature_dependence(tempurature=np.linspace(150, 500), stable_resistance=330000, p_0=-0.00420717061486765 * (55000/330000), stable_tempurature=298))
plt.legend()

# T = 175
plt.figure(1)
lrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=175+273, stable_resistance=9e4, p_0=0.000801158151673717 * (2400/9e4), stable_tempurature=298)
hrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=175+273, stable_resistance=330000, p_0=-0.00420717061486765 * (55000/330000), stable_tempurature=298)
lrs_model_parameters.update({'p_0': np.log10(lrs_stable_resistance)})
hrs_model_parameters.update({'p_0': np.log10(hrs_stable_resistance)})
plt.scatter(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), label='LRS Model T=175$^{\circ}$C', color='b', marker='^')
plt.scatter(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), label='HRS Model T=175$^{\circ}$C', color='r', marker='^')

# T = 75
lrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=75+273, stable_resistance=9e4, p_0=0.000801158151673717 * (2400/9e4), stable_tempurature=298)
hrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=75+273, stable_resistance=330000, p_0=-0.00420717061486765 * (55000/330000), stable_tempurature=298)
lrs_model_parameters.update({'p_0': np.log10(lrs_stable_resistance)})
hrs_model_parameters.update({'p_0': np.log10(hrs_stable_resistance)})
plt.scatter(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), label='LRS Model T=75$^{\circ}$C', color='b', marker='o')
plt.scatter(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), label='HRS Model T=75$^{\circ}$C', color='r', marker='o')

handles, labels = h.gca().get_legend_handles_labels()
order = [0, 1, 4, 5, 2, 3]
h.gca().legend([handles[idx] for idx in order], [labels[idx] for idx in order])
plt.show()
