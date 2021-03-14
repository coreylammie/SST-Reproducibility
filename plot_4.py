import numpy as np
import pandas as pd
import math
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Import and concatenate experimental data
u_10 = pd.read_csv('Experimental Data/4_10uA.csv')
u_20 = pd.read_csv('Experimental Data/4_20uA.csv')
u_50 = pd.read_csv('Experimental Data/4_50uA.csv')
min=1.4
max=2.4
scale_input = interp1d([min, max], [0, 1])
x = 1.8
y = 1e5
k = math.log10(y) / (1 - (2 * scale_input(x) - 1) ** (2))
def window(input):
    m = interp1d([min, max], [0, 1])
    scaled_input = scale_input(input)
    return 10 ** (k * (1 - (2 * scaled_input - 1) ** (2)))

# Plot the experimental data and results from the model
x = np.linspace(1.4, 2.4, 100)
y = window(x)
plt.figure(1)
label_size = 20
tick_size = 16
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
plt.plot(x, y, 'k', linewidth=2)
plt.plot(u_10.iloc[:, 0], u_10.iloc[:, 1], linestyle='', marker='o', markersize=10, markerfacecolor='None', markeredgewidth=2, color='g')
plt.plot(u_20.iloc[:, 0], u_20.iloc[:, 1], linestyle='', marker='s', markersize=10, markerfacecolor='None', markeredgewidth=2, color='r')
plt.plot(u_50.iloc[:, 0], u_50.iloc[:, 1], linestyle='', marker='^', markersize=10, markerfacecolor='None', markeredgewidth=2, color='b')
plt.yscale('log')
plt.ylim(ymin=1e0, ymax=1e6)
plt.gca().set_axisbelow(True)
plt.minorticks_on()
plt.grid(b=True, which='both')
plt.xlabel('$V_{stop}$', fontsize=label_size)
plt.ylabel('$e_{th}$', fontsize=label_size)
plt.gca().tick_params(axis='both', which='major', labelsize=tick_size)
plt.gca().tick_params(axis='both', which='minor', labelsize=tick_size)
plt.show()
