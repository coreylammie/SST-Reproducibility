import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


hrs = pd.read_csv('HRS_raw_data.csv')
lrs = pd.read_csv('LRS_raw_data.csv')

plt.figure(1)
plt.xlim(1e2, 1e8)
plt.ylim(1e4, 12e4)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS', color='r')
plt.xscale('log')
plt.xlabel('Time (s)')
plt.ylabel('R ($\Omega$)')
plt.grid(b=True, axis='both')
plt.minorticks_on()
plt.show()
