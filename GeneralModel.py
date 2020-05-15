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

    def model_tempurature_dependence(self, tempurature, stable_resistance, p_0, stable_tempurature=273):
        return np.piecewise(tempurature, [tempurature <= stable_tempurature, tempurature > stable_tempurature], [stable_resistance, lambda tempurature: 10 ** (p_0 * tempurature + np.log10(stable_resistance) - p_0 * stable_tempurature)])

    def fit_tempurature(self, raw_data_x, raw_data_y, stable_resistance, stable_tempurature=273, r_on=True):
        parameters = Parameters()
        parameters.add('stable_resistance', value=stable_resistance, vary=False)
        parameters.add('p_0', value=1)
        parameters.add('stable_tempurature', value=stable_tempurature, vary=False)
        model = Model(self.model_tempurature_dependence)
        fitted_model = model.fit(raw_data_y, tempurature=raw_data_x, params=parameters)
        return fitted_model.best_values

    def model(self, input, p_0, p_1, p_2, p_3, cell_size=None):
        assert input is not None and len(input) > 0, 'input is invalid.'
        assert input.ndim == 1, 'input must be 1-dimensional.'
        output = np.zeros(input.shape)
        if cell_size is None:
            cell_size = 10

        if self.operation_mode == OperationMode.gradual:
            threshold = p_1 * np.exp(p_2 * cell_size)
            return np.piecewise(input, [input <= threshold, input > threshold], [10 ** p_0, lambda input: 10 ** (p_3 * cell_size * np.log10(input) + np.log10(10 ** p_0) - p_3 * cell_size * np.log10(threshold))])
        elif self.operation_mode == OperationMode.sudden:
            threshold = cell_size * p_2 + p_3
            return np.piecewise(input, [input < threshold, input >= threshold], [10 ** pq_0, lambda input: 10 ** pq_1])

    def objective(self, parameters, raw_data_x, raw_data_y):
        assert len(raw_data_x) == len(raw_data_y)
        concatenated_output = np.array([])
        concatenated_model_output = np.array([])
        for i in range(len(raw_data_x)):
            concatenated_output = np.append(concatenated_output, raw_data_y[i].flatten()).flatten()
            model_output = self.model_gradual_convergence(raw_data_x[i], parameters['pq_0'], parameters['pq_1'], parameters['pq_2_%d' % (i + 1)], parameters['pq_3'], cell_size=parameters['cell_size_%d' % (i + 1)])
            concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()

        return np.abs(concatenated_model_output - concatenated_output)

    def fit(self, cell_size=None, **kwargs):
        if self.operation_mode == OperationMode.gradual:
            assert {'raw_data_x', 'raw_data_y', 'stable_resistance', 'threshold'} <= set(kwargs)
        elif self.operation_mode == OperationMode.sudden
            assert {'initial_resistance', 'stable_resistance', 'threshold'} <= set(kwargs)

        self.fit.__globals__.update(kwargs)
        if self.cell_size_dependance == False:
            cell_size = 10
        else:
            assert cell_size is not None
            if self.operation_mode == OperationMode.gradual:
                if type(cell_size) is list:
                    assert len(raw_data_x) == len(raw_data_y) == len(threshold) == len(cell_size)
                else:
                    raw_data_x = [raw_data_x]
                    raw_data_y = [raw_data_y]
                    threshold = [threshold]
                    cell_size = [cell_size]
            elif self.operation_mode == OperationMode.sudden:
                if type(cell_size) is list:
                    assert len(threshold) == len(cell_size)
                else:
                    threshold = [threshold]
                    cell_size = [cell_size]

        threshold_slope = 1
        threshold_intercept = 0
        if type(cell_size) is list:
            if len(cell_size) > 1:
                linear_model = lmfit.models.LinearModel()
                fitted_linear_model = linear_model.fit(threshold, x=cell_size)
                threshold_slope = fitted_linear_model.best_values['slope']
                threshold_intercept = fitted_linear_model.best_values['intercept']
                def det_threshold(cell_size):
                    return fitted_linear_model.best_values['slope'] * cell_size + fitted_linear_model.best_values['intercept']

                for index, cell in enumerate(cell_size):
                    threshold[index] = det_threshold(cell)

        if self.operation_mode == OperationMode.gradual:
            parameters = Parameters()
            parameters.add('p_0', value=np.log10(stable_resistance), vary=False)
            parameters.add('p_1', value=0.5)
            parameters.add('p_3', value=0)
            for i in range(len(cell_size)):
                parameters.add('threshold_%d' % (i + 1), value=threshold[i], vary=False)
                parameters.add('cell_size_%d' % (i + 1), value=cell_size[i], vary=False)
                if i == 0:
                    parameters.add('p_2_%d' % (i + 1), value=0.5, expr='log(threshold_1 / p_1) / cell_size_1')
                else:
                    parameters.add('p_2_%d' % (i + 1), value=0.5, expr='log(threshold_%d / p_1) / cell_size_%d and p_2_1' % (i + 1, i + 1))

            out = minimize(self.objective, parameters, args=(raw_data_x, raw_data_y)) # print(report_fit(out.params))
            return {'p_0': out.params['p_0'], 'p_1': out.params['p_1'], 'p_2': out.params['p_2_1'], 'p_3': out.params['p_3']}
        elif self.operation_mode == OperationMode.sudden:
            return {'p_0': np.log10(initial_resistance), 'p_1': np.log10(stable_resistance), 'p_2': threshold_slope, 'p_3': threshold_intercept}
