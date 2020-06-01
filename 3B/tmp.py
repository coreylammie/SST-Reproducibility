import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from GeneralModel import GeneralModel
from GeneralModel import OperationMode


matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rcParams['font.family'] = 'sans-serif'
label_size = 16
tick_size = 12
h = plt.figure(1)
plt.xlabel('Time (s)', fontsize=label_size)
plt.ylabel('CIFAR-10 Test Set Accuracy (%)', fontsize=label_size)

plt.show()
