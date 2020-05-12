import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Currently hardcoded for 10
lrs = pd.read_csv('LRS_10_raw_data.csv')
hrs = pd.read_csv('HRS_10_raw_data.csv')

plt.figure(1)
plt.xlim(1e2, 1e9)
plt.ylim(1e3, 1e6)
plt.scatter(lrs.iloc[:, 0].values, lrs.iloc[:, 1].values, label='LRS', color='b')
plt.scatter(hrs.iloc[:, 0].values, hrs.iloc[:, 1].values, label='HRS', color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Cycles')
plt.ylabel('Resistance ($\Omega$)')
plt.legend()
plt.grid(b=True, axis='both')
plt.minorticks_on()
plt.show()
