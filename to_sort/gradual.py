import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lengths = [10, 20, 30]
for index, length in enumerate(lengths):
    plt.figure(index)
    lrs = pd.read_csv('gradual_%d_lrs.csv' % length)
    hrs = pd.read_csv('gradual_%d_hrs.csv' % length)
    lrs = lrs.sort_values(by=['x'], ascending=True)
    hrs = hrs.sort_values(by=['x'], ascending=True)
    plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], markeredgecolor='red', marker='s', markersize=20, markerfacecolor='w', linestyle='-', color='red')
    plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], markeredgecolor='blue', marker='s', markersize=20, markerfacecolor='w', linestyle='-', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.xlim([1e2, 1e9])
    plt.ylim([1e3, 1e6])
    plt.xlabel('Cycle Number')
    plt.ylabel('Resistance ($\Omega$)')
    plt.title('Gradual Failure ($d$ = %dnm)' % length)

plt.show()
