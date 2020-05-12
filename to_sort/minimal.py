from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lmfit
from lmfit import minimize, Parameters, Model

hrs_10 = pd.read_csv('gradual_10_hrs.csv')
lrs_10 = pd.read_csv('gradual_10_lrs.csv')
x_lrs = np.array(lrs_10.iloc[:, 0])
y_lrs = np.array(lrs_10.iloc[:, 1])
x_hrs = np.array(hrs_10.iloc[:, 0])
y_hrs = np.array(hrs_10.iloc[:, 1])

def piecewise_linear(x, x0, k1, k2, k3):
    return np.piecewise(x, [x < x0, x >= x0], [lambda x:k1, lambda x:k1 + k2 * (x - x0) ** k3])

# LRS
params_lrs = Parameters()
params_lrs.add('x0', value=1e3, min=0)
params_lrs.add('k1', value=1, min=0)
params_lrs.add('k2', value=1)
params_lrs.add('k3', value=1)
model_lrs = Model(piecewise_linear)
result_lrs = model_lrs.fit(y_lrs, x=x_lrs, params=params_lrs)
print(result_lrs.fit_report())

# HRS
params_hrs = Parameters()
params_hrs.add('x0', value=1e3, min=0)
params_hrs.add('k1', value=1, min=0)
params_hrs.add('k2', value=1)
params_hrs.add('k3', value=1)
model_hrs = Model(piecewise_linear)
result_hrs = model_hrs.fit(y_hrs, x=x_hrs, params=params_hrs)
print(result_hrs.fit_report())

plt.figure(1)
# plt.plot(x_lrs, y_lrs, 'o', color='blue')
plt.plot(x_lrs, result_lrs.best_fit, "s", color='blue')
# plt.plot(x_hrs, y_hrs, "o", color='red')
plt.plot(x_hrs, result_hrs.best_fit, "s", color='red')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.xlim([1e2, 1e9])
plt.ylim([1e3, 1e6])
plt.xlabel('Cycle Number')
plt.ylabel('Resistance ($\Omega$)')
plt.show()
