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

    def gradual_convergence(self, input, input_stable, pq_0, pq_1, pq_2, pq_3, r_on=True, cell_size=None):
        assert input is not None and len(input) > 0, 'input is invalid.'
        assert input.ndim == 1, 'input must be 1-dimensional.'
        output = np.zeros(input.shape)
        if cell_size is None:
            cell_size = 10

        input_stable = pq_1 * np.exp(pq_2 * cell_size)
        if r_on:
            return np.piecewise(input, [input <= input_stable, input > input_stable], [lambda input: 10 ** pq_0, lambda input: 10 ** (pq_3 * cell_size * np.log10(input) + np.log10(10 ** pq_0) + pq_3 * np.log10(pq_1 * np.exp(pq_2 * cell_size)))])
        else:
            return np.piecewise(input, [input <= input_stable, input > input_stable], [lambda input: 10 ** pq_0, lambda input: 10 ** (pq_3 * cell_size * np.log10(input) + np.log10(10 ** pq_0) - pq_3 * np.log10(pq_1 * np.exp(pq_2 * cell_size)))])

    def fit(self, raw_data, operation_mode):
        assert operation_mode == OperationMode.gradual, 'Sudden convergence is not currently supported.'
        fitting_parameters = Parameters()
        fitting_parameters.add('pq_0', value=1)
        fitting_parameters.add('pq_1', value=1)
        fitting_parameters.add('pq_2', value=1)
        fitting_parameters.add('pq_3', value=1)
        model = Model(self.gradual_convergence)
        raw_data_x = raw_data[:, 0]
        raw_data_y = raw_data[:, 1]
        fitted_model = model.fit(raw_data_y, x=raw_data_x, params=fitting_parameters, cell_size=cell_size)
        print(fitted_model.fit_report)
        # TODO implement modular fit routine for various cell sizes
