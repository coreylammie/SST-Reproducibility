import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


lengths = [10, 20, 30]
for index, length in enumerate(lengths):
    lrs = pd.read_csv('LRS_%d_raw_data.csv' % length)
    hrs = pd.read_csv('HRS_%d_raw_data.csv' % length)
    plt.figure(index + 1)
    plt.grid()
    plt.xlim(1e2, 1e9)
    plt.ylim(1e3, 1e6)
    plt.scatter(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, label='LRS', color='b')
    plt.scatter(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, label='HRS', color='r')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Device length = %dnm' % length)
    plt.xlabel('Cycles')
    plt.ylabel('Resistance ($\Omega$)')
    plt.legend()

plt.show()
