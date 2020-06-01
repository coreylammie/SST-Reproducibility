import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


lrs = pd.read_csv('LRS_raw_data.csv')
lrs = lrs.sort_values(by=lrs.columns[0])
hrs = pd.read_csv('HRS_raw_data.csv')
hrs = hrs.sort_values(by=hrs.columns[0])

matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
plt.figure(1)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlim(1e2, 1e8)
plt.ylim(1e4, 12e4)
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS', linestyle='-', color='b', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS', linestyle='-', color='r', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
plt.xscale('log')
plt.xlabel('Cycle Number', fontsize=label_size)
plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
