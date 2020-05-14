import numpy as np
import pandas as pd
import math
import enum
from enum import Enum, auto
import scipy
from scipy import optimize
import lmfit
from lmfit import minimize, Parameters, Model, report_fit
import matplotlib.pyplot as plt


class OperationMode(Enum):
    sudden = auto()
    gradual = auto()


class GeneralModel():
    def __init__(self, operation_mode, tempurature_dependance=False, cell_size_dependance=False):
        self.operation_mode = operation_mode
        self.tempurature_dependance = tempurature_dependance
        self.cell_size_dependance = cell_size_dependance

    def model_tempurature_dependance(self, tempurature, p_0, p_1, p_2):
        return p_0 * (p_1 ** tempurature) + p_2

    def fit_tempurature(self, raw_data_x, raw_data_y, r_on=True):
        parameters = Parameters()
        if r_on:
            parameters.add('p_0', value=1, min=0)
        else:
            parameters.add('p_0', value=-1, max=0)
        # parameters.add('p_0', value=0)
        parameters.add('p_1', value=2, min=1)
        parameters.add('p_2', value=1, min=0)
        model = Model(self.model_tempurature_dependance)
        fitted_model = model.fit(raw_data_y, tempurature=raw_data_x, params=parameters)
        return fitted_model.best_values


    def model_gradual_convergence(self, input, pq_0, pq_1, pq_2, pq_3, cell_size=None):
        assert input is not None and len(input) > 0, 'input is invalid.'
        assert input.ndim == 1, 'input must be 1-dimensional.'
        output = np.zeros(input.shape)
        if cell_size is None:
            cell_size = 10

        input_stable = pq_1 * np.exp(pq_2 * cell_size)
        return np.piecewise(input, [input <= input_stable, input > input_stable], [lambda input: 10 ** pq_0, lambda input: 10 ** (pq_3 * cell_size * np.log10(input) + np.log10(10 ** pq_0) - pq_3 * cell_size * np.log10(input_stable))])

    def objective(self, parameters, raw_data_x, raw_data_y):
        assert len(raw_data_x) == len(raw_data_y)
        concatenated_output = np.array([])
        concatenated_model_output = np.array([])
        for i in range(len(raw_data_x)):
            concatenated_output = np.append(concatenated_output, raw_data_y[i].flatten()).flatten()
            model_output = self.gradual_convergence(raw_data_x[i], parameters['pq_0'], parameters['pq_1'], parameters['pq_2_%d' % (i + 1)], parameters['pq_3'], cell_size=parameters['cell_size_%d' % (i + 1)])
            concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()

        return np.abs(concatenated_model_output - concatenated_output)

    def fit(self, raw_data_x, raw_data_y, stable_resistance, threshold, cell_size=None):
        assert self.operation_mode == OperationMode.gradual, 'Sudden convergence is not currently supported.'
        if self.cell_size_dependance == False:
            cell_size = 10
        else:
            assert cell_size is not None
            if isinstance(cell_size, list):
                assert len(raw_data_x) == len(raw_data_y) == len(threshold) == len(cell_size)
            else:
                raw_data_x = [raw_data_x]
                raw_data_y = [raw_data_y]
                threshold = [threshold]
                cell_size = [cell_size]

        if len(cell_size) > 1:
            linear_model = lmfit.models.LinearModel()
            fitted_linear_model = linear_model.fit(threshold, x=cell_size)
            def det_threshold(cell_size):
                return fitted_linear_model.best_values['slope'] * cell_size + fitted_linear_model.best_values['intercept']

            for index, cell in enumerate(cell_size):
                threshold[index] = det_threshold(cell)

        parameters = Parameters()
        parameters.add('pq_0', value=np.log10(stable_resistance), vary=False)
        parameters.add('pq_1', value=0.5)
        parameters.add('pq_3', value=0)
        for i in range(len(cell_size)):
            parameters.add('threshold_%d' % (i + 1), value=threshold[i], vary=False)
            parameters.add('cell_size_%d' % (i + 1), value=cell_size[i], vary=False)

            if i == 0:
                parameters.add('pq_2_%d' % (i + 1), value=0.5, expr='log(threshold_1 / pq_1) / cell_size_1')
            else:
                parameters.add('pq_2_%d' % (i + 1), value=0.5, expr='log(threshold_%d / pq_1) / cell_size_%d and pq_2_1' % (i + 1, i + 1))


        out = minimize(self.objective, parameters, args=(raw_data_x, raw_data_y))
        print(report_fit(out.params))
        return out.params
