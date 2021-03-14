import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import scipy.interpolate


def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

# Import and concatenate experimental data
phi_R = pd.read_csv('Experimental Data/3D_phi.csv', header=None)
phi_R_x = np.array(phi_R.iloc[:, 0].values)
phi_R_y = np.array(phi_R.iloc[:, 1].values)
data = pd.read_csv('Experimental Data/3D_R.csv', header=None)
R = data.iloc[:, 0].values
phi = log_interp1d(phi_R_x, phi_R_y)(R)
threshold = data.iloc[:, 1].values
# Plot the experimental data and results from the model
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.plot(phi, threshold, linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.xscale('log')
plt.yscale('log')
f_ = lambda cell_size, p_0, p_1: p_0 * np.exp(p_1 * cell_size)
model_estimate = f_(phi * 1e9, 172.8, 0.258)
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 20
tick_size = 16
plt.title('Au/NiO/Si', fontsize=label_size)
plt.plot(phi, model_estimate, linestyle='--', color='blue', marker='o', markersize=15, markerfacecolor='None', markeredgewidth=1)
plt.xlabel('$\phi$ (m)', fontsize=label_size)
plt.ylabel('$τ_R$ (s)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.gca().tick_params(axis='both', which='minor', labelsize=tick_size)
plt.grid(b=True, which='both')
plt.show()
