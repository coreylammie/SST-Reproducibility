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
    # TODO Add optimization method argument to __init__
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
        assert input is not None
        input = np.array(input)
        output = np.zeros(input.shape)
        if cell_size is None:
            cell_size = 10

        if self.operation_mode == OperationMode.gradual:
            threshold = p_1 * np.exp(p_2 * cell_size)
            return np.piecewise(input, [input <= threshold, input > threshold], [10 ** p_0, lambda input: 10 ** (p_3 * cell_size * np.log10(input) + np.log10(10 ** p_0) - p_3 * cell_size * np.log10(threshold))])
        elif self.operation_mode == OperationMode.sudden:
            threshold =  p_2 * np.exp(p_3 * cell_size)
            return np.piecewise(input, [input <= threshold, input > threshold], [10 ** p_0, lambda input: 10 ** p_1])

    def objective(self, parameters, raw_data_x, raw_data_y):
        assert len(raw_data_x) == len(raw_data_y)
        concatenated_output = np.array([])
        concatenated_model_output = np.array([])
        number_of_sets = int(parameters['number_of_sets'])
        if number_of_sets == 1:
            concatenated_output = np.append(concatenated_output, raw_data_y).flatten()
            model_output = self.model(raw_data_x, parameters['p_0'], parameters['p_1'], parameters['p_2_1'], parameters['p_3'], cell_size=parameters['cell_size_1'])
            concatenated_model_output = np.append(concatenated_model_output, model_output).flatten()
        else:
            for i in range(number_of_sets):
                concatenated_output = np.append(concatenated_output, raw_data_y[i].flatten()).flatten()
                model_output = self.model(raw_data_x[i], parameters['p_0'], parameters['p_1'], parameters['p_2_%d' % (i + 1)], parameters['p_3'], cell_size=parameters['cell_size_%d' % (i + 1)])
                concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()

        return np.abs(concatenated_model_output - concatenated_output)

    def fit(self, cell_size=None, **kwargs):
        if self.operation_mode == OperationMode.gradual:
            assert {'raw_data_x', 'raw_data_y', 'stable_resistance', 'threshold'} <= set(kwargs)
            raw_data_x = kwargs['raw_data_x']
            raw_data_y = kwargs['raw_data_y']
        elif self.operation_mode == OperationMode.sudden:
            assert {'initial_resistance', 'stable_resistance', 'threshold'} <= set(kwargs)
            initial_resistance = kwargs['initial_resistance']

        stable_resistance = kwargs['stable_resistance']
        threshold = kwargs['threshold']
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

        if self.operation_mode == OperationMode.gradual:
            parameters = Parameters()
            parameters.add('p_0', value=np.log10(stable_resistance), vary=False)
            parameters.add('p_1', value=0.5)
            parameters.add('p_3', value=0)
            if not type(cell_size) is list:
                cell_size = [cell_size]
                threshold = [threshold]

            parameters.add('number_of_sets', value=len(cell_size), vary=False)
            for i in range(len(cell_size)):
                parameters.add('threshold_%d' % (i + 1), value=threshold[i], vary=False)
                parameters.add('cell_size_%d' % (i + 1), value=cell_size[i], vary=False)
                if i == 0:
                    parameters.add('p_2_%d' % (i + 1), value=0.5, expr='log(threshold_1 / p_1) / cell_size_1')
                else:
                    parameters.add('p_2_%d' % (i + 1), value=0.5, expr='log(threshold_%d / p_1) / cell_size_%d and p_2_1' % (i + 1, i + 1))

            out = minimize(self.objective, parameters, args=(raw_data_x, raw_data_y), max_nfev=100000)
            print(out.params.pretty_print())
            print(out.residual)
            return {'p_0': out.params['p_0'], 'p_1': out.params['p_1'], 'p_2': out.params['p_2_1'], 'p_3': out.params['p_3']}
        elif self.operation_mode == OperationMode.sudden:
            threshold_model = Model(lambda cell_size, p_2, p_3: p_2 * np.exp(p_3 * cell_size))
            parameters = Parameters()
            if type(cell_size) is list:
                parameters.add('p_2', value=0.5)
            else:
                parameters.add('p_2', value=0.5, vary=False)

            parameters.add('p_3', value=0.5)
            out = threshold_model.fit(threshold, cell_size=cell_size, params=parameters)
            return {'p_0': np.log10(initial_resistance), 'p_1': np.log10(stable_resistance), 'p_2': out.params['p_2'], 'p_3':  out.params['p_3']}
