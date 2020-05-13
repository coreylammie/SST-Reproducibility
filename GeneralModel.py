import numpy as np
import pandas as pd
import math
import enum
from enum import Enum, auto
import scipy
from scipy import optimize
import lmfit
from lmfit import minimize, Parameters, Model


class OperationMode(Enum):
    sudden = auto()
    gradual = auto()


class GeneralModel():
    def __init__(self, operation_mode, tempurature_dependance=False, cell_size_dependance=False):
        self.operation_mode = operation_mode
        self.tempurature_dependance = tempurature_dependance
        self.cell_size_dependance = cell_size_dependance

    def gradual_convergence(self, input, pq_0, pq_1, pq_2, pq_3, r_off=True, cell_size=None):
        assert input is not None and len(input) > 0, 'input is invalid.'
        assert input.ndim == 1, 'input must be 1-dimensional.'
        output = np.zeros(input.shape)
        if cell_size is None:
            cell_size = 10

        input_stable = pq_1 * np.exp(pq_2 * cell_size)
        return np.piecewise(input, [input <= input_stable, input > input_stable], [lambda input: 10 ** pq_0, lambda input: 10 ** (pq_3 * cell_size * np.log10(input) + np.log10(10 ** pq_0) - pq_3 * cell_size * np.log10(input_stable))])


    def cell_size_harness(self, raw_data_x_nd, raw_data_y_nd, threshold_list, cell_size_list):
        assert cell_size_list is not None
        out = np.zeros(raw_data_y_nd.shape)
        print(out.shape)
        exit(0)
        # for index, cell_size in enumerate(cell_size_list):




    def fit(self, raw_data_x, raw_data_y, stable_resistance, threshold, cell_size=None):
        assert self.operation_mode == OperationMode.gradual, 'Sudden convergence is not currently supported.'
        if self.cell_size_dependance == False:
            cell_size = 10
        else:
            assert cell_size is not None and raw_data_x.shape[1] == raw_data_y.shape[1] == len(list(stable_resistance)) == len(list(threshold)) == len(list(cell_size))
            stable_resistance = list(stable_resistance)
            threshold = list(threshold)
            cell_size = list(cell_size)

        parameters = Parameters()
        parameters.add('threshold', value=threshold, vary=False)
        parameters.add('cell_size', value=cell_size, vary=False)
        parameters.add('pq_0', value=np.log10(stable_resistance), vary=False)
        parameters.add('pq_1', value=0.5)
        parameters.add('pq_2', value=0.5, expr='log(threshold / pq_1) / cell_size')
        parameters.add('pq_3', value=0.5)
        if self.cell_size_dependance:
            model = Model(cell_size_harness(self.gradual_convergence))
        else:
            model = Model(self.gradual_convergence)

        exit(0)
        # fitted_model = model.fit(raw_data_y, input=raw_data_x, params=parameters)
        # print(fitted_model.fit_report()))
        # return fitted_model.best_values


# return np.concatenate(out).ravel()
