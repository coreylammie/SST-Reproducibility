import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


fit_raw_data = True
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/2C_LRS_raw_data.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/2C_HRS_raw_data.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
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

# Plot the experimental data and results from the fitted models
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.title('TiN/HfO$_x$/TiN', fontsize=label_size)
plt.xlim(4e1, 2e6)
plt.ylim(5e4, 5e5)
plt.xscale('log')
plt.yscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Time (s)', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
if fit_raw_data:
    general_model = GeneralModel(operation_mode=None)
    lrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=175+273, stable_resistance=9e4, p_0=0.000801158151673717 * (2400/9e4), stable_tempurature=298)
    hrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=175+273, stable_resistance=330000, p_0=-0.00420717061486765 * (55000/330000), stable_tempurature=298)
    lrs_model_parameters.update({'p_0': np.log10(lrs_stable_resistance)})
    hrs_model_parameters.update({'p_0': np.log10(hrs_stable_resistance)})
    plt.plot(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), linestyle='--', color='b', marker='^', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), linestyle='--', color='r', marker='^', markersize=15, markerfacecolor='None', markeredgewidth=1)
    lrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=75+273, stable_resistance=9e4, p_0=0.000801158151673717 * (2400/9e4), stable_tempurature=298)
    hrs_stable_resistance = general_model.model_tempurature_dependence(tempurature=75+273, stable_resistance=330000, p_0=-0.00420717061486765 * (55000/330000), stable_tempurature=298)
    lrs_model_parameters.update({'p_0': np.log10(lrs_stable_resistance)})
    hrs_model_parameters.update({'p_0': np.log10(hrs_stable_resistance)})
    plt.plot(lrs.iloc[:, 0].values, lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters), linestyle='-.', color='b', marker='v', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters), linestyle='-.', color='r', marker='v', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.axvspan(1200, 1e6, facecolor='b', alpha=0.25)
plt.axvline(x=1200)
plt.axvline(x=1e6)
plt.show()
