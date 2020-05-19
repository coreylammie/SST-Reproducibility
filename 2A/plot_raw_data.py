import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


lengths = [10, 20, 30]
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
for index, length in enumerate(lengths):
    lrs = pd.read_csv('LRS_%d_raw_data.csv' % length)
    lrs = lrs.sort_values(by=lrs.columns[0])
    hrs = pd.read_csv('HRS_%d_raw_data.csv' % length)
    hrs = hrs.sort_values(by=hrs.columns[0])
    plt.figure(index + 1)
    plt.gca().set_axisbelow(True)
    plt.minorticks_on()
    plt.grid(b=True, which='both')
    plt.xlim(1e2, 1e9)
    plt.ylim(1e3, 1e6)
    plt.plot(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, label='LRS', color='b', linestyle='-', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.plot(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, label='HRS', color='r', linestyle='-', marker='s', markersize=17.5, markerfacecolor='None', markeredgewidth=2.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('(%dnm x %dnm)' % (length, length), fontsize=label_size)
    plt.xlabel('Cycle Number', fontsize=label_size)
    plt.ylabel('Resistance ($\Omega$)', fontsize=label_size)
    plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)

plt.show()
