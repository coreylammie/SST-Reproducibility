import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


lrs = pd.read_csv('sudden_lrs.csv')
hrs = pd.read_csv('sudden_hrs.csv')

lrs = lrs.sort_values(by=['x'], ascending=True)
hrs = hrs.sort_values(by=['x'], ascending=True)


plt.figure(1)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], 's', markeredgecolor='r', markerfacecolor=(1,1,1,0), markersize=15, markeredgewidth=2, linestyle='-', linewidth=2, color='r')
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], 's', markeredgecolor='b', markerfacecolor=(1,1,1,0), markersize=15, markeredgewidth=2, linestyle='-', linewidth=2, color='b')
plt.xscale('log')
plt.grid()
plt.xlim([1e2, 1e8])
plt.ylim([0, 15e4])
plt.xlabel('Cycle Number')
plt.ylabel('Resistance ($\Omega$)')
plt.title('Sudden Failure')
plt.show()


# plt.plot(x_lrs[index][np.argsort(x_lrs[index])], y_lrs[index][np.argsort(x_lrs[index])], 's', markeredgecolor='b', markerfacecolor=(1,1,1,0), markersize=15, markeredgewidth=2, linestyle='-', linewidth=2, color='b')
# plt.plot(x_lrs[index][np.argsort(x_lrs[index])], piecewise_linear(x_lrs[index][np.argsort(x_lrs[index])], device_length=device_length, pos=True, **result_lrs.best_values), 'o', markeredgecolor='b', markerfacecolor=(1,1,1,0), markersize=15, markeredgewidth=1, linestyle='--', linewidth=1, color='b')
