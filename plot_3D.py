import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import scipy as sp
import scipy.interpolate
import lmfit
from lmfit import minimize, Parameters, Model, fit_report


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

plt.figure(1)
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
plt.show()
