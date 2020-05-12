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
x0s = [10 ** 4, 10 ** 7, 10 ** 9]

def fitme(x, device_lengths, k1, k2, k3, pos=True):
    out = copy.deepcopy(x)
    for index, device_length in enumerate(device_lengths):
        x_split = x0s[index]
        x_ = x[index]
        if pos:
            out[index] = np.piecewise(x_, [x_ < x_split, x_ >= x_split], [lambda x_:k1, lambda x_:k1 + k1 * (x_ - x_split) ** (k2)])
        else:
            out[index] = np.piecewise(x_, [x_ < x_split, x_ >= x_split], [lambda x_:k1, lambda x_:k1 - (k1 * (x_ - x_split) ** (k2))])

    return np.concatenate(out).ravel()

def piecewise_linear(x, device_length, k1, k2, k3, pos=True):
    x_split = x0s[list(device_lengths).index(device_length)]
    if pos:
        return np.piecewise(x, [x < x_split, x >= x_split], [lambda x:k1, lambda x:k1 + k1 * (x - x_split) ** (k2)])
    else:
        return np.piecewise(x, [x < x_split, x >= x_split], [lambda x:k1, lambda x:k1 - (k1 * (x - x_split) ** (k2))])

# LRS
params_lrs = Parameters()
params_lrs.add('k1', value=1, min=0)
params_lrs.add('k2', value=0)
params_lrs.add('k3', value=0)
model_lrs = Model(fitme, independent_vars=['x', 'device_lengths'])
result_lrs = model_lrs.fit(np.concatenate(y_lrs).ravel(), x=x_lrs, device_lengths=device_lengths, params=params_lrs, pos=True)

# HRS
params_hrs = Parameters()
params_hrs.add('k1', value=1, min=0)
params_hrs.add('k2', value=0)
params_hrs.add('k3', value=0)
model_hrs = Model(fitme, independent_vars=['x', 'device_lengths'])
result_hrs = model_hrs.fit(np.concatenate(y_hrs).ravel(), x=x_hrs, device_lengths=device_lengths, params=params_hrs, pos=True)

for index, device_length in enumerate(device_lengths):
    # if device_length == 10:
    #     x_interp = np.logspace(2, 8, 1e2)
    # else:
    x_interp = np.logspace(2, 9, 1e2)

    plt.figure(index)
    plt.plot(x_lrs[index], y_lrs[index], 'o', color='blue')
    plt.plot(x_interp, piecewise_linear(x_interp, device_length=device_length, pos=True, **result_lrs.best_values), 's', color='blue')
    plt.plot(x_hrs[index], y_hrs[index], 'o', color='red')
    plt.plot(x_interp, piecewise_linear(x_interp, device_length=device_length, pos=False, **result_hrs.best_values), 's', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.xlim([1e2, 1e9])
    plt.ylim([1e3, 1e6])
    plt.xlabel('Cycle Number')
    plt.ylabel('Resistance ($\Omega$)')
    plt.title('Gradual Failure ($d$ = %dnm)' % device_length)

plt.show()
