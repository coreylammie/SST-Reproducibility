import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


hrs = pd.read_csv('HRS.csv')
lrs = pd.read_csv('LRS.csv')

plt.figure(1)
plt.xlim(1e0, 1e7)
plt.ylim(1e4, 1e6)
plt.grid()
plt.plot(hrs.iloc[:, 0], hrs.iloc[:, 1], linestyle='', marker='.', color='b', markersize=10)
plt.plot(lrs.iloc[:, 0], lrs.iloc[:, 1], linestyle='', marker='.', color='r', markersize=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('R ($\Omega$)')


plt.show()
