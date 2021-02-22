import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from GeneralModel import GeneralModel
from GeneralModel import OperationMode
import scipy as sp
import scipy.interpolate
import lmfit
from lmfit import minimize, Parameters, Model, fit_report

fit_raw_data = True
plt.figure(1, figsize=(20, 9))
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12

plt.subplot(2, 4, 1)
# Import and concatenate experimental data
lrs_10 = pd.read_csv('Experimental Data/2A_LRS_10.csv')
lrs_10 = lrs_10.sort_values(by=lrs_10.columns[0])
lrs_20 = pd.read_csv('Experimental Data/2A_LRS_20.csv')
lrs_20 = lrs_20.sort_values(by=lrs_20.columns[0])
hrs_10 = pd.read_csv('Experimental Data/2A_HRS_10.csv')
hrs_10 = hrs_10.sort_values(by=hrs_10.columns[0])
hrs_20 = pd.read_csv('Experimental Data/2A_HRS_20.csv')
hrs_20 = hrs_20.sort_values(by=hrs_20.columns[0])
lrs = [lrs_10, lrs_20]
hrs = [hrs_10, hrs_20]
cell_sizes = [10, 20]

# Fit the model in gradual operation mode to the 10nm and 20nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_raw_data_x = {}
    lrs_raw_data_x[(10, None)] = lrs_10.iloc[:, 0].values
    lrs_raw_data_x[(20, None)] = lrs_20.iloc[:, 0].values
    lrs_raw_data_y = {}
    lrs_raw_data_y[(10, None)] = lrs_10.iloc[:, 1].values
    lrs_raw_data_y[(20, None)] = lrs_20.iloc[:, 1].values
    lrs_threshold = {}
    lrs_threshold[(10, None)] = 1e4
    lrs_threshold[(20, None)] = 1e7
    lrs_model_parameters = lrs_model.fit(raw_data_x=lrs_raw_data_x,
                               raw_data_y=lrs_raw_data_y,
                               initial_resistance=4400,
                               threshold=lrs_threshold,
                               cell_size=cell_sizes)
    hrs_raw_data_x = {}
    hrs_raw_data_x[(10, None)] = hrs_10.iloc[:, 0].values
    hrs_raw_data_x[(20, None)] = hrs_20.iloc[:, 0].values
    hrs_raw_data_y = {}
    hrs_raw_data_y[(10, None)] = hrs_10.iloc[:, 1].values
    hrs_raw_data_y[(20, None)] = hrs_20.iloc[:, 1].values
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 1e4
    hrs_threshold[(20, None)] = 1e7
    hrs_model_parameters = hrs_model.fit(raw_data_x=hrs_raw_data_x,
                               raw_data_y=hrs_raw_data_y,
                               initial_resistance=65000,
                               threshold=hrs_threshold,
                               cell_size=cell_sizes)
# Plot the experimental data and results from the fitted models


plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('TiN/Hf(Al)O/Hf/TiN', fontsize=label_size)
for i in range(len(cell_sizes)):
    plt.grid(b=True, which='both')
    plt.xlim(1e2, 1e9)
    plt.ylim(1e3, 1e6)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(hrs[i].iloc[:, 0].values, hrs[i].iloc[:, 1].values, linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(hrs[i].iloc[:, 0].values, hrs_model.model(hrs[i].iloc[:, 0].values, **hrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Cycle Number', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.subplot(2, 4, 2)
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/2B_LRS.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/2B_HRS.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=False)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=False)
    lrs_model_parameters = {'initial_resistance': 750, 'p_1': 0.5, 'p_2': 0., 'p_3': 0.}
    hrs_model_parameters = {'initial_resistance': 9e10, 'p_1': 0.5, 'p_2': 0.5, 'p_3': -0.12}

# Plot the experimental data and results from the fitted models
plt.title('Cu/HfO$_x$/Pt', fontsize=label_size)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlim(1e1, 1e8)
plt.ylim(1e2, 1e12)
plt.xscale('log')
plt.yscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(lrs.iloc[:, 0].values, lrs_model.model(hrs.iloc[:, 0].values, **lrs_model_parameters, cell_size=None), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters, cell_size=None), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Cycle Number', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.subplot(2, 4, 3)
# Import and concatenate experimental data
lrs_20 = pd.read_csv('Experimental Data/2C_LRS_20.csv', header=None)
lrs_20 = lrs_20.sort_values(by=lrs_20.columns[0])
lrs_30 = pd.read_csv('Experimental Data/2C_LRS_30.csv', header=None)
lrs_30 = lrs_30.sort_values(by=lrs_30.columns[0])
lrs_40 = pd.read_csv('Experimental Data/2C_LRS_40.csv', header=None)
lrs_40 = lrs_40.sort_values(by=lrs_40.columns[0])
hrs_20 = pd.read_csv('Experimental Data/2C_HRS_20.csv', header=None)
hrs_20 = hrs_20.sort_values(by=hrs_20.columns[0])
hrs_30 = pd.read_csv('Experimental Data/2C_HRS_30.csv', header=None)
hrs_30 = hrs_30.sort_values(by=hrs_30.columns[0])
hrs_40 = pd.read_csv('Experimental Data/2C_HRS_40.csv', header=None)
hrs_40 = hrs_40.sort_values(by=hrs_40.columns[0])
lrs = [lrs_20, lrs_30, lrs_40]
hrs = [hrs_20, hrs_30, hrs_40]
cell_sizes = [20, 30, 40]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=True)
    hrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=True)
    lrs_raw_data_x = {}
    lrs_raw_data_x[(20, None)] = lrs_20.iloc[:, 0].values
    lrs_raw_data_x[(30, None)] = lrs_30.iloc[:, 0].values
    lrs_raw_data_x[(40, None)] = lrs_40.iloc[:, 0].values
    lrs_raw_data_y = {}
    lrs_raw_data_y[(20, None)] = lrs_20.iloc[:, 1].values
    lrs_raw_data_y[(30, None)] = lrs_30.iloc[:, 1].values
    lrs_raw_data_y[(40, None)] = lrs_40.iloc[:, 1].values
    lrs_threshold = {}
    lrs_threshold[(20, None)] = 2.4e6
    lrs_threshold[(30, None)] = 2e7
    lrs_threshold[(40, None)] = 2.5e8
    lrs_model_parameters = lrs_model.fit(raw_data_x=lrs_raw_data_x,
                               raw_data_y=lrs_raw_data_y,
                               initial_resistance=14000,
                               stable_resistance=5e7,
                               threshold=lrs_threshold,
                               cell_size=cell_sizes)
    hrs_raw_data_x = {}
    hrs_raw_data_x[(20, None)] = hrs_20.iloc[:, 0].values
    hrs_raw_data_x[(30, None)] = hrs_30.iloc[:, 0].values
    hrs_raw_data_x[(40, None)] = hrs_40.iloc[:, 0].values
    hrs_raw_data_y = {}
    hrs_raw_data_y[(20, None)] = hrs_20.iloc[:, 1].values
    hrs_raw_data_y[(30, None)] = hrs_30.iloc[:, 1].values
    hrs_raw_data_y[(40, None)] = hrs_40.iloc[:, 1].values
    hrs_threshold = {}
    hrs_threshold[(20, None)] = 2e6
    hrs_threshold[(30, None)] = 1.5e7
    hrs_threshold[(40, None)] = 1.7e8
    hrs_model_parameters = hrs_model.fit(raw_data_x=hrs_raw_data_x,
                               raw_data_y=hrs_raw_data_y,
                               initial_resistance=300000,
                               stable_resistance=5e7,
                               threshold=hrs_threshold,
                               cell_size=cell_sizes)

plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('TiN/Hf(Al)O/Hf/TiN', fontsize=label_size)
for i in range(len(cell_sizes)):
    plt.grid(b=True, which='both')
    plt.xlim(1e0, 1e9)
    plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(hrs[i].iloc[:, 0].values, hrs[i].iloc[:, 1].values, linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, **lrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(hrs[i].iloc[:, 0].values, hrs_model.model(hrs[i].iloc[:, 0].values, **hrs_model_parameters, cell_size=cell_sizes[i]), linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Cycle Number', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.subplot(2, 4, 4)
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/2D_LRS.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/2D_HRS.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=False)
    hrs_model = GeneralModel(operation_mode=OperationMode.sudden, cell_size_dependance=False)
    lrs_threshold = {}
    lrs_threshold[(10, None)] = 0.
    lrs_model_parameters = lrs_model.fit(initial_resistance=2.00e4, stable_resistance=2.00e4, threshold=lrs_threshold)
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 2.227e7
    hrs_model_parameters = hrs_model.fit(initial_resistance=10.75e4, stable_resistance=2.00e4, threshold=hrs_threshold)
    lrs_model_output = lrs_model.model(lrs.iloc[:, 0].values, **lrs_model_parameters)
    hrs_model_output = hrs_model.model(hrs.iloc[:, 0].values, **hrs_model_parameters)

# Plot the experimental data and results from the fitted models
plt.title('TiN/ETML/HfO$_x$/TiN', fontsize=label_size)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlim(1e2, 1e8)
plt.ylim(1e4, 12e4)
plt.xscale('log')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
if fit_raw_data:
    plt.plot(lrs.iloc[:, 0].values, lrs_model_output, linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
    plt.plot(hrs.iloc[:, 0].values, hrs_model_output, linestyle='--', color='r', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

plt.xlabel('Cycle Number', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.subplot(2, 4, 5)
# Import and concatenate experimental data
lrs_200 = pd.read_csv('Experimental Data/3A_LRS_200.csv', header=None)
lrs_200 = lrs_200.sort_values(by=lrs_200.columns[0])
lrs_225 = pd.read_csv('Experimental Data/3A_LRS_225.csv', header=None)
lrs_225 = lrs_225.sort_values(by=lrs_225.columns[0])
lrs_250 = pd.read_csv('Experimental Data/3A_LRS_250.csv', header=None)
lrs_250 = lrs_250.sort_values(by=lrs_250.columns[0])
lrs = [lrs_250, lrs_225, lrs_200]
tempuratures = [200+273, 225+273, 250+273]
# tempuratures = [200, 225, 250]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.sudden)
    initial_resistance = 40000
    stable_resistance = 2.6e7
    tempurature_threshold = 298
    lrs_model_parameters = {'initial_resistance': initial_resistance, 'p_1': np.log10(stable_resistance), 'p_2': 0.003117, 'p_3': 0.05626*298/10, 'tempurature_threshold': tempurature_threshold}

plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('Pt/Cu:MoO$_x$/GdO$_x$/Pt', fontsize=label_size)
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    plt.xlim(1e0, 1e4)
    plt.ylim(1e4, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs[i].iloc[:, 0].values, lrs[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs[i].iloc[:, 0].values, lrs_model.model(lrs[i].iloc[:, 0].values, tempurature=tempuratures[i], **lrs_model_parameters), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.subplot(2, 4, 6)
# Import and concatenate experimental data
lrs_200_O = pd.read_csv('Experimental Data/3B_LRS_O_200.csv', header=None)
lrs_200_O = lrs_200_O.sort_values(by=lrs_200_O.columns[0])
lrs_250_O = pd.read_csv('Experimental Data/3B_LRS_O_250.csv', header=None)
lrs_220_O = lrs_250_O.sort_values(by=lrs_250_O.columns[0])
lrs_300_O = pd.read_csv('Experimental Data/3B_LRS_O_300.csv', header=None)
lrs_300_O = lrs_300_O.sort_values(by=lrs_300_O.columns[0])
lrs_O = [lrs_200_O, lrs_250_O, lrs_300_O]
lrs_200_A = pd.read_csv('Experimental Data/3B_LRS_Al_200.csv', header=None)
lrs_200_A = lrs_200_A.sort_values(by=lrs_200_A.columns[0])
lrs_250_A = pd.read_csv('Experimental Data/3B_LRS_Al_250.csv', header=None)
lrs_220_A = lrs_250_A.sort_values(by=lrs_250_A.columns[0])
lrs_300_A = pd.read_csv('Experimental Data/3B_LRS_Al_300.csv', header=None)
lrs_300_A = lrs_300_A.sort_values(by=lrs_300_A.columns[0])
lrs_A = [lrs_200_A, lrs_250_A, lrs_300_A]
tempuratures = [273+200, 273+250, 273+300]

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_O_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_O_model_parameters = {'initial_resistance': 4250, 'p_1': 14610000000, 'p_2': -1.9784220000000001, 'p_3': 0.14041884744983046, 'tempurature_threshold': 298}
    lrs_A_model = GeneralModel(operation_mode=OperationMode.gradual, cell_size_dependance=True)
    lrs_A_model_parameters = {'initial_resistance': 4250, 'p_1': 790100000000, 'p_2': -2.5029019999999997, 'p_3': 0.0577995548620432257, 'tempurature_threshold': 298}

plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.title('Ti/HfO$_x$/TiN\nTi/HfAlO/TiN', fontsize=label_size)
for i in range(len(tempuratures)):
    plt.grid(b=True, which='both')
    # plt.xlim(1e0, 1e9)
    # plt.ylim(5e3, 1e7)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(lrs_O[i].iloc[:, 0].values, lrs_O[i].iloc[:, 1].values, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(lrs_A[i].iloc[:, 0].values, lrs_A[i].iloc[:, 1].values, linestyle='-', color='magenta', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    if fit_raw_data:
        plt.plot(lrs_O[i].iloc[:, 0].values, lrs_O_model.model(lrs_O[i].iloc[:, 0].values, **lrs_O_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='b', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
        plt.plot(lrs_A[i].iloc[:, 0].values, lrs_A_model.model(lrs_A[i].iloc[:, 0].values, **lrs_A_model_parameters, tempurature=tempuratures[i]), linestyle='--', color='magenta', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)

    plt.xlabel('Time (s)', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.tight_layout()
plt.subplot(2, 4, 7)
# Import and concatenate experimental data
lrs = pd.read_csv('Experimental Data/3C_LRS.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('Experimental Data/3C_HRS.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

# Fit the model in gradual operation mode to the 20nm and 30nm experimental data
if fit_raw_data:
    lrs_model = GeneralModel(operation_mode=OperationMode.gradual, tempurature_dependance=True, cell_size_dependance=False)
    lrs_raw_data_x = {}
    lrs_raw_data_x[(10, None)] = lrs.iloc[:, 0].values
    lrs_raw_data_y = {}
    lrs_raw_data_y[(10, None)] = lrs.iloc[:, 1].values
    lrs_threshold = {}
    lrs_threshold[(10, None)] = 1e3
    lrs_model_parameters = lrs_model.fit(raw_data_x=lrs_raw_data_x,
                                         raw_data_y=lrs_raw_data_y,
                                         initial_resistance=8.8e4,
                                         threshold=lrs_threshold)
    hrs_model = GeneralModel(operation_mode=OperationMode.gradual, tempurature_dependance=True, cell_size_dependance=False)
    hrs_raw_data_x = {}
    hrs_raw_data_x[(10, None)] = hrs.iloc[:, 0].values
    hrs_raw_data_y = {}
    hrs_raw_data_y[(10, None)] = hrs.iloc[:, 1].values
    hrs_threshold = {}
    hrs_threshold[(10, None)] = 1e3
    hrs_model_parameters = hrs_model.fit(raw_data_x=hrs_raw_data_x,
                                         raw_data_y=hrs_raw_data_y,
                                         initial_resistance=330000,
                                         threshold=hrs_threshold)

# Plot the experimental data and results from the fitted models
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
plt.axvspan(1200, 1e6, facecolor='b', alpha=0.25)
plt.axvline(x=1200)
plt.axvline(x=1e6)

plt.tight_layout()
plt.subplot(2, 4, 8)

def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

phi_R = pd.read_csv('Experimental Data/3D_phi.csv', header=None)
phi_R_x = np.array(phi_R.iloc[:, 0].values)
phi_R_y = np.array(phi_R.iloc[:, 1].values)

data = pd.read_csv('Experimental Data/3D_R.csv', header=None)
R = data.iloc[:, 0].values
phi = log_interp1d(phi_R_x, phi_R_y)(R)
threshold = data.iloc[:, 1].values

plt.gca().set_axisbelow(True)
plt.minorticks_on()

plt.plot(phi, threshold, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
# plt.xlim(4e-9, 2e-8)
# plt.ylim([3e2, 2e4])
plt.xscale('log')
plt.yscale('log')


f_ = lambda cell_size, p_1, p_2: p_1 * np.exp(p_2 * cell_size)

# print(phi)
# print(threshold)
model_estimate = f_(phi * 1e9, 172.8, 0.258)

matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12

plt.title('Au/NiO/Si', fontsize=label_size)

plt.plot(phi, model_estimate, linestyle='--', color='blue', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.xlabel('$\phi$ (m)', fontsize=label_size)
plt.ylabel('$τ_R$ (s)', fontsize=label_size)
plt.grid(b=True, which='both')

plt.tight_layout()
plt.show()
