import numpy as np
import pandas as pd
import math
import enum
from enum import Enum, auto
import scipy
from scipy import optimize
import lmfit
from lmfit import minimize, Parameters, Model, fit_report
import matplotlib.pyplot as plt

# np.seterr(all='raise')


class OperationMode(Enum):
    sudden = auto()
    gradual = auto()

class GeneralModel():
    # TODO Add optimization method argument to __init__
    def __init__(self, operation_mode, tempurature_dependance=False, cell_size_dependance=False):
        self.operation_mode = operation_mode

    # def model_tempurature_dependence(self, tempurature, stable_resistance, p_0, stable_tempurature=273):
    #     return np.piecewise(tempurature, [tempurature <= stable_tempurature, tempurature > stable_tempurature], [stable_resistance, lambda tempurature: 10 ** (p_0 * tempurature + np.log10(stable_resistance) - p_0 * stable_tempurature)])
    #
    # def fit_tempurature_resistance(self, raw_data_x, raw_data_y, stable_resistance, stable_tempurature=273):
    #     parameters = Parameters()
    #     parameters.add('stable_resistance', value=stable_resistance, vary=False)
    #     parameters.add('p_0', value=1)
    #     parameters.add('stable_tempurature', value=stable_tempurature, vary=False)
    #     model = Model(self.model_tempurature_dependence)
    #     fitted_model = model.fit(raw_data_y, tempurature=raw_data_x, params=parameters)
    #     return fitted_model.best_values

    def model(self, input, initial_resistance, p_1, p_2, p_3, cell_size=None, tempurature=None, tempurature_threshold=None):
        assert input is not None
        input = np.array(input)
        output = np.zeros(input.shape)
        if cell_size is None or cell_size <= 0:
            cell_size = 10

        if tempurature is not None:
            assert tempurature_threshold is not None
            tempurature_constant = tempurature / tempurature_threshold
        else:
            tempurature_constant = 1

        p_0 = np.log10(initial_resistance)
        if self.operation_mode == OperationMode.gradual:
            threshold = p_1 * np.exp(p_2 * cell_size * tempurature_constant)
            # print(threshold)
            out = np.piecewise(input, [input <= threshold, input > threshold], [10 ** p_0, lambda input: 10 ** ((p_3 * cell_size) ** tempurature_constant * np.log10(input) + np.log10(10 ** p_0) - (p_3 * cell_size) ** tempurature_constant * np.log10(threshold))])
            # out = np.piecewise(input, [input <= threshold, input > threshold], [10 ** p_0, lambda input: 10 ** (p_3 * cell_size * tempurature_constant * np.log10(input) + np.log10(10 ** p_0) - p_3 * cell_size * tempurature_constant * np.log10(threshold))])
            if out is None or np.isnan(out).any() or np.isinf(out).any():
                print('NaN Encountered')
                print('out:', out)
                print('input:', input)
                print('threshold: %f' % threshold)
                print('p_0: %f' % p_0)
                print('p_3: %f' % p_3)
                print('tempurature_constant: %f' % tempurature_constant)
                exit(0)
            else:
                # print(out)
                return out

        elif self.operation_mode == OperationMode.sudden:
            threshold =  p_2 * np.exp(p_3 * cell_size * tempurature_constant)
            return np.piecewise(input, [input <= threshold, input > threshold], [10 ** p_0, lambda input: 10 ** p_1])

    def objective(self, parameters, raw_data_x, raw_data_y):
        assert len(raw_data_x) == len(raw_data_y)
        concatenated_output = np.array([])
        concatenated_model_output = np.array([])
        if int(parameters['cell_size_sets']) * int(parameters['tempurature_sets']) == 1:
            concatenated_output = np.append(concatenated_output, raw_data_y[(None, parameters['tempurature_1'])]).flatten()
            model_output = self.model(raw_data_x[(None, parameters['tempurature_1'])], parameters['initial_resistance'], parameters['p_1'], parameters['p_2_1_1'], parameters['p_3'], cell_size=parameters['cell_size_1'],
                                      tempurature=parameters['tempurature_1'], tempurature_threshold=parameters['tempurature_threshold'])
            concatenated_model_output = np.append(concatenated_model_output, model_output).flatten()
        else:
            for i in range(int(parameters['cell_size_sets'])):
                if int(parameters['tempurature_sets']) > 1:
                    for j in range(int(parameters['tempurature_sets'])):
                        concatenated_output = np.append(concatenated_output, raw_data_y[(parameters['cell_size_%d' % (i+1)].value, parameters['tempurature_%d' % (j+1)].value)]).flatten()
                        model_output = self.model(raw_data_x[(parameters['cell_size_%d' % (i+1)].value, parameters['tempurature_%d' % (j+1)].value)], parameters['initial_resistance'], parameters['p_1'],
                                                  parameters['p_2_1_1'], parameters['p_3'], cell_size=parameters['cell_size_%d' % (i+1)],
                                                  tempurature=parameters['tempurature_%d' % (j+1)], tempurature_threshold=parameters['tempurature_threshold'])
                        concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()
                else:
                    concatenated_output = np.append(concatenated_output, raw_data_y[(parameters['cell_size_%d' % (i+1)].value, None)]).flatten()
                    model_output = self.model(raw_data_x[(parameters['cell_size_%d' % (i+1)].value, None)], parameters['initial_resistance'], parameters['p_1'],
                                              parameters['p_2_1_1'], parameters['p_3'], cell_size=parameters['cell_size_%d' % (i+1)])
                    concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()

        return np.abs(concatenated_model_output - concatenated_output)

    def fit(self, cell_size=None, tempurature=None, tempurature_threshold=None, **kwargs):
        if self.operation_mode == OperationMode.gradual:
            assert {'raw_data_x', 'raw_data_y', 'initial_resistance', 'threshold'} <= set(kwargs)
            raw_data_x = kwargs['raw_data_x']
            raw_data_y = kwargs['raw_data_y']
        elif self.operation_mode == OperationMode.sudden:
            assert {'initial_resistance', 'stable_resistance', 'threshold'} <= set(kwargs)
            stable_resistance = kwargs['stable_resistance']

        initial_resistance = kwargs['initial_resistance']
        threshold = kwargs['threshold']
        if cell_size is not None:
            if type(cell_size) is list:
                assert len(threshold) == len(cell_size)
        else:
            cell_size = [10]

        if tempurature is not None:
            assert tempurature_threshold is not None
            if type(tempurature) is not list:
                tempurature = [tempurature]

        assert type(threshold) is dict
        if self.operation_mode == OperationMode.gradual:
            if tempurature is not None:
                f_ = lambda tempurature_threshold, tempurature, cell_size, p_1, p_2: p_1 * np.exp(p_2 * cell_size * (tempurature / tempurature_threshold))
                threshold_model = Model(f_, independent_vars=['tempurature','cell_size'])
            else:
                f_ = lambda cell_size, p_1, p_2: p_1 * np.exp(p_2 * cell_size)
                threshold_model = Model(f_)

            parameters = Parameters()
            if len(cell_size) == 1:
                parameters.add('p_1', value=0.1, vary=False)
            else:
                parameters.add('p_1', value=0.1)

            parameters.add('p_2', value=0.1)
            if tempurature is not None:
                parameters.add('tempurature_threshold', value=tempurature_threshold, vary=False)
                threshold_ = np.empty((len(tempurature), len(cell_size)))
                for i_idx, tempurature_ in enumerate(tempurature):
                    for j_idx, cell_size_ in enumerate(cell_size):
                        threshold_[i_idx, j_idx] = threshold[(cell_size_, tempurature_)]

                out = threshold_model.fit(threshold_, parameters, tempurature=tempurature, cell_size=cell_size)
            else:
                threshold_ = np.empty(len(cell_size))
                for i_idx, cell_size_ in enumerate(cell_size):
                    threshold_[i_idx] = threshold[(cell_size_, None)]

                out = threshold_model.fit(threshold_, parameters, cell_size=cell_size)
                # print(out)
                # print(fit_report(out))

            parameters = Parameters()
            parameters.add('initial_resistance', value=initial_resistance, vary=False)
            # parameters.add('p_1', value=5.471e+19, vary=False)
            parameters.add('p_1', value=out.params['p_1'], vary=False)
            # parameters.add('p_2_1_1', value= -0.07368 * 298 / 10, vary=False)
            parameters.add('p_2_1_1', value=out.params['p_2'], vary=False)
            parameters.add('p_3', value=0.5)
            parameters.add('tempurature_threshold', value=tempurature_threshold, vary=False)
            if tempurature is not None:
                parameters.add('tempurature_sets', value=len(tempurature), vary=False)
                for i in range(len(tempurature)):
                    parameters.add('tempurature_%d' % (i+1), value=tempurature[i], vary=False)
            else:
                parameters.add('tempurature_sets', value=0, vary=False)

            parameters.add('cell_size_sets', value=len(cell_size), vary=False)
            for i in range(len(cell_size)):
                parameters.add('cell_size_%d' % (i+1), value=cell_size[i], vary=False)
                # if tempurature is not None:
                #     for j in range(len(tempurature)):
                #         # parameters.add('threshold_%d_%d' % (i+1, j+1), value=threshold[(cell_size[i], tempurature[j])], vary=False)
                #         # if i == 0 and j == 0:
                #         #     parameters.add('p_2_1_%d' % (j+1), value=0.5, expr='log(threshold_1_1 / p_1) * tempurature_threshold / (cell_size_1 * tempurature_1)')
                #         #     # parameters.add('p_2_1_%d' % (j+1), value=0.5, expr='log(threshold_1_%d / p_1) / (cell_size_1 * min(tempurature_threshold / tempurature_%d, 1))' % (j+1, j+1))
                #         # else:
                #         #     parameters.add('p_2_%d_%d' % (i+1, j+1), value=0.5, expr='p_2_1_1 and (log(threshold_%d_%d / p_1) * tempurature_threshold / (cell_size_%d * tempurature_%d)) ' % (i+1, j+1, i+1, j+1))
                #         #     # parameters.add('p_2_%d_%d' % (i+1, j+1), value=0.5, expr='log(threshold_%d_%d / p_1) / (cell_size_%d * min(tempurature_threshold / tempurature_%d, 1)) and p_2_1_1' % (i+1, j+1, i+1, j+1))
                #     parameters.add('p_2_1_1', value=1e-12, min=1e-12, expr='ln(threshold_1_1 / p_1) * tempurature_threshold / (cell_size_1 * tempurature_1) == ln(threshold_1_2 / p_1) * tempurature_threshold / (cell_size_1 * tempurature_2) == ln(threshold_1_3 / p_1) * tempurature_threshold / (cell_size_1 * tempurature_3)')
                # else:
                #     parameters.add('threshold_%d' % (i+1), value=threshold[(cell_size[i], None)], vary=False)
                #     parameters.add('cell_size_%d' % (i+1), value=cell_size[i], vary=False)
                #     if i == 0:
                #         parameters.add('p_2_1_1', value=0.5, expr='log(threshold_1 / p_1) / cell_size_1')
                #     else:
                #         parameters.add('p_2_1_%d' % (i+1), value=0.5, expr='log(threshold_%d / p_1) / cell_size_%d and p_2_1_1' % (i+1, i+1))

            out = minimize(self.objective, parameters, args=(raw_data_x, raw_data_y))
            # print(fit_report(out))
            # print(out.params)


            return {'initial_resistance': out.params['initial_resistance'], 'p_1': out.params['p_1'], 'p_2': out.params['p_2_1_1'], 'p_3': out.params['p_3'], 'tempurature_threshold': out.params['tempurature_threshold']}
        elif self.operation_mode == OperationMode.sudden:
            if tempurature is not None:
                threshold_model = Model(lambda tempurature_threshold, tempurature, cell_size, p_2, p_3: p_2 * np.exp(p_3 * cell_size * (tempurature / tempurature_threshold)), independent_vars=['tempurature','cell_size'])
            else:
                threshold_model = Model(lambda cell_size, p_2, p_3: p_2 * np.exp(p_3 * cell_size))

            parameters = Parameters()
            if len(cell_size) == 1:
                parameters.add('p_2', value=0.1, vary=False)
            else:
                parameters.add('p_2', value=0.1)

            parameters.add('p_3', value=0.1)
            if tempurature is not None:
                threshold_ = np.empty((len(tempurature), len(cell_size)))
                for i_idx, tempurature_ in enumerate(tempurature):
                    for j_idx, cell_size_ in enumerate(cell_size):
                        threshold_[i_idx, j_idx] = threshold[(cell_size_, tempurature_)]
            else:
                threshold_ = np.empty(len(cell_size))
                for i_idx, cell_size_ in enumerate(cell_size):
                    threshold_[i_idx] = threshold[(cell_size_, None)]

            out = threshold_model.fit(threshold_, parameters, cell_size=cell_size)

            if tempurature is not None:
                return {'initial_resistance': initial_resistance, 'p_1': np.log10(stable_resistance), 'p_2': out.params['p_2'], 'p_3':  out.params['p_3'], 'tempurature_threshold': out.params['tempurature_threshold']}
            else:
                return {'initial_resistance': initial_resistance, 'p_1': np.log10(stable_resistance), 'p_2': out.params['p_2'], 'p_3':  out.params['p_3'], 'tempurature_threshold': None}
