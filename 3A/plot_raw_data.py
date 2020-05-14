import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


hrs = pd.read_csv('HRS_raw_data.csv')
lrs = pd.read_csv('LRS_raw_data.csv')

plt.figure(1)
plt.grid()
plt.xlim(200, 450)
plt.ylim(1e3, 1e5)
plt.scatter(lrs.iloc[:, 0], lrs.iloc[:, 1], label='LRS', color='b')
plt.scatter(hrs.iloc[:, 0], hrs.iloc[:, 1], label='HRS', color='r')
plt.yscale('log')
plt.xlabel('Tempurature (K)')
plt.ylabel('Resistance ($\Omega$)')
plt.show()
