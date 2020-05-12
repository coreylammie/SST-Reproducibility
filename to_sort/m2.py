from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit
from lmfit import minimize, Parameters, Model
import copy


hrs_10 = pd.read_csv('gradual_10_hrs.csv')
lrs_10 = pd.read_csv('gradual_10_lrs.csv')
hrs_20 = pd.read_csv('gradual_20_hrs.csv')
lrs_20 = pd.read_csv('gradual_20_lrs.csv')
hrs_30 = pd.read_csv('gradual_30_hrs.csv')
lrs_30 = pd.read_csv('gradual_30_lrs.csv')

x_lrs = np.array([np.array(lrs_10.iloc[:, 0]), np.array(lrs_20.iloc[:, 0]), np.array(lrs_30.iloc[:, 0])], dtype=object)
y_lrs = np.array([np.array(lrs_10.iloc[:, 1]), np.array(lrs_20.iloc[:, 1]), np.array(lrs_30.iloc[:, 1])], dtype=object)
x_hrs = np.array([np.array(hrs_10.iloc[:, 0]), np.array(hrs_20.iloc[:, 0]), np.array(hrs_30.iloc[:, 0])], dtype=object)
y_hrs = np.array([np.array(hrs_10.iloc[:, 1]), np.array(hrs_20.iloc[:, 1]), np.array(hrs_30.iloc[:, 1])], dtype=object)

device_lengths = [10, 20, 30]
x0s = [10 ** 3, 10 ** 4, 10 ** 9]

def piecewise_linear(x, device_length, k1, k2, k3):
    x_split = x0s[device_lengths.index(device_length)]
    return np.piecewise(x, [x < x_split, x >= x_split], [lambda x:k1, lambda x:k1 + k2 * (x - x_split) ** k3])

x_interp = np.logspace(2, 9, 1e2)
for index, device_length in enumerate(device_lengths):
    plt.figure(index)

    # LRS
    params_lrs = Parameters()
    params_lrs.add('k1', value=1, min=0)
    params_lrs.add('k2', value=1)
    params_lrs.add('k3', value=1)
    model_lrs = Model(piecewise_linear, independent_vars=['x', 'device_length'])
    result_lrs = model_lrs.fit(y_lrs[index], x=x_lrs[index], device_length=device_lengths[index], params=params_lrs)
    # print(result_lrs.fit_report())

    # HRS
    params_hrs = Parameters()
    params_hrs.add('k1', value=1, min=0)
    params_hrs.add('k2', value=-1)
    params_hrs.add('k3', value=1)
    model_hrs = Model(piecewise_linear, independent_vars=['x', 'device_length'])
    result_hrs = model_hrs.fit(y_hrs[index], x=x_hrs[index], device_length=device_lengths[index], params=params_hrs)
    # print(result_hrs.fit_report())

    plt.plot(x_lrs[index], y_lrs[index], 'o', color='blue')
    plt.plot(x_interp, piecewise_linear(x_interp, device_length=device_length, **result_lrs.best_values), 's', color='blue')
    plt.plot(x_hrs[index], y_hrs[index], 'o', color='red')
    plt.plot(x_interp, piecewise_linear(x_interp, device_length=device_length, **result_hrs.best_values), 's', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.xlim([1e2, 1e9])
    plt.ylim([1e3, 1e6])
    plt.xlabel('Cycle Number')
    plt.ylabel('Resistance ($\Omega$)')
    plt.title('Gradual Failure ($d$ = %dnm)' % device_length)

plt.show()
