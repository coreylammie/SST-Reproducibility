import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as optimization

lrs = pd.read_csv('gradual_20_lrs.csv')
hrs = pd.read_csv('gradual_20_hrs.csv')

def r_on_objective_function(n, n_stable, p_0, p_1, p_2):
    r_on_init = 4400
    v = 10
    r_max = lrs.iloc[:, 1].max()
    out = np.zeros(n.shape)
    out[n <= n_stable] = r_on_init
    out[n > n_stable] = r_on_init + v * n[n > n_stable] ** p_0 + p_1 * n[n > n_stable]
    return out

def r_off_objective_function(n, n_stable, p_0, p_1, p_2):
    r_off_init = 50000
    v = 10
    r_max = lrs.iloc[:, 1].max()
    out = np.zeros(n.shape)
    out[n <= n_stable] = r_off_init
    out[n > n_stable] = r_off_init - v * n[n > n_stable] ** p_0 + p_1 * n[n > n_stable]
    return out

r_on_opt, r_on_fit = optimization.curve_fit(r_on_objective_function, lrs.iloc[:, 0], lrs.iloc[:, 1], [1e3, 0.5, 1, 1])
r_off_opt, r_off_fit = optimization.curve_fit(r_off_objective_function, hrs.iloc[:, 0], hrs.iloc[:, 1], [10e4, 0.5, 1, 1])

plt.figure(1)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.xlim([1e2, 1e9])
plt.ylim([1e3, 1e6])
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], markeredgecolor='blue', marker='s', markersize=20, markerfacecolor='w', linestyle='', color='blue')
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], markeredgecolor='red', marker='s', markersize=20, markerfacecolor='w', linestyle='', color='red')

plt.plot(lrs.iloc[:, 0], r_on_objective_function(lrs.iloc[:, 0], r_on_opt[0], r_on_opt[1], r_on_opt[2], r_on_opt[3]), markeredgecolor='blue', marker='o', markersize=20, markerfacecolor='w', linestyle='', color='blue')
plt.plot(hrs.iloc[:, 0], r_off_objective_function(hrs.iloc[:, 0], r_off_opt[0], r_off_opt[1], r_off_opt[2], r_off_opt[3]), markeredgecolor='red', marker='o', markersize=20, markerfacecolor='w', linestyle='', color='red')
plt.show()
