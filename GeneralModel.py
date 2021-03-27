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


class OperationMode(Enum):
    sudden = auto()
    gradual = auto()

class GeneralModel():
    def __init__(self, operation_mode, temperature_dependance=False, cell_size_dependance=False):
        self.operation_mode = operation_mode
        self.temperature_dependance = temperature_dependance
        self.cell_size_dependance = cell_size_dependance

    def model(self, input, initial_resistance, p_0, p_1, p_2, p_3=0., stable_resistance=None, cell_size=None, temperature=None, temperature_threshold=None):
        assert input is not None
        input = np.array(input)
        output = np.zeros(input.shape)
        if cell_size is None or cell_size <= 0:
            cell_size = 10

        if temperature is not None:
            assert temperature_threshold is not None
            temperature_constant = temperature_threshold / temperature
        else:
            temperature_constant = 1

        if self.operation_mode == OperationMode.gradual:
            threshold =  p_0 * np.exp(p_1 * cell_size + p_2 * temperature_constant)
            return np.piecewise(input, [input <= threshold, input > threshold], [initial_resistance, lambda input: 10 ** (p_3 * (p_1 * cell_size + p_2 * temperature_constant) * np.log10(input) + np.log10(initial_resistance) - p_3 * (p_1 * cell_size + p_2 * temperature_constant) * np.log10(threshold))])

        elif self.operation_mode == OperationMode.sudden:
            threshold =  p_0 * np.exp(p_1 * cell_size + p_2 * temperature_constant)
            return np.piecewise(input, [input <= threshold, input > threshold], [initial_resistance, stable_resistance])

    def objective(self, parameters, raw_data_x, raw_data_y):
        assert len(raw_data_x) == len(raw_data_y)
        concatenated_output = np.array([])
        concatenated_model_output = np.array([])
        if int(parameters['cell_size_sets']) * int(parameters['temperature_sets']) == 1:
            concatenated_output = np.append(concatenated_output, raw_data_y[(None, parameters['temperature_1'])]).flatten()
            model_output = self.model(raw_data_x[(None, parameters['temperature_1'])], parameters['initial_resistance'], parameters['p_0'], parameters['p_1'], parameters['p_2'], parameters['p_3'], cell_size=parameters['cell_size_1'],
                                      temperature=parameters['temperature_1'], temperature_threshold=parameters['temperature_threshold'])
            concatenated_model_output = np.append(concatenated_model_output, model_output).flatten()
        else:
            for i in range(int(parameters['cell_size_sets'])):
                if int(parameters['temperature_sets']) > 1:
                    for j in range(int(parameters['temperature_sets'])):
                        concatenated_output = np.append(concatenated_output, raw_data_y[(parameters['cell_size_%d' % (i+1)].value, parameters['temperature_%d' % (j+1)].value)]).flatten()
                        model_output = self.model(raw_data_x[(parameters['cell_size_%d' % (i+1)].value, parameters['temperature_%d' % (j+1)].value)], parameters['initial_resistance'].value, parameters['p_0'].value, parameters['p_1'].value, parameters['p_2'].value, parameters['p_3'].value, cell_size=parameters['cell_size_%d' % (i+1)].value, temperature=parameters['temperature_%d' % (j+1)], temperature_threshold=parameters['temperature_threshold'])
                        concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()
                else:
                    concatenated_output = np.append(concatenated_output, raw_data_y[(parameters['cell_size_%d' % (i+1)].value, None)]).flatten()
                    k = raw_data_x[(parameters['cell_size_%d' % (i+1)].value, None)]
                    model_output = self.model(raw_data_x[(parameters['cell_size_%d' % (i+1)].value, None)], parameters['initial_resistance'].value, parameters['p_0'].value, parameters['p_1'].value, parameters['p_2'].value, parameters['p_3'].value, cell_size=parameters['cell_size_%d' % (i+1)].value)
                    concatenated_model_output = np.append(concatenated_model_output, model_output.flatten()).flatten()

        return np.abs(concatenated_model_output - concatenated_output)

    def fit(self, cell_size=None, temperature=None, temperature_threshold=None, **kwargs):
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

        if temperature is not None:
            assert temperature_threshold is not None
            if type(temperature) is not list:
                temperature = [temperature]

        assert type(threshold) is dict
        if temperature is not None:
            f_ = lambda p_0, p_1, p_2, temperature_threshold, temperature, cell_size: p_0 * np.exp(p_1 * cell_size + p_2 * (temperature_threshold / temperature))
            threshold_model = Model(f_, independent_vars=['temperature','cell_size'])
        else:
            f_ = lambda p_0, p_1, cell_size: p_0 * np.exp(p_1 * cell_size)
            threshold_model = Model(f_, independent_vars=['cell_size'])

        parameters = Parameters()
        if len(cell_size) == 1:
            parameters.add('p_0', value=0.1, vary=False)
        else:
            parameters.add('p_0', value=0.1)

        parameters.add('p_1', value=0.1)
        if temperature is not None:
            parameters.add('p_2', value=0.1)
            parameters.add('temperature_threshold', value=temperature_threshold, vary=False)
            threshold_ = np.empty((len(temperature), len(cell_size)))
            for i_idx, temperature_ in enumerate(temperature):
                for j_idx, cell_size_ in enumerate(cell_size):
                    threshold_[i_idx, j_idx] = threshold[(cell_size_, temperature_)]

            out = threshold_model.fit(threshold_, parameters, temperature=temperature, cell_size=cell_size)
        else:
            threshold_ = np.empty(len(cell_size))
            for i_idx, cell_size_ in enumerate(cell_size):
                threshold_[i_idx] = threshold[(cell_size_, None)]

            out = threshold_model.fit(threshold_, parameters, cell_size=cell_size)

        if self.operation_mode == OperationMode.gradual:
            parameters = Parameters()
            parameters.add('initial_resistance', value=initial_resistance, vary=False)
            parameters.add('p_0', value=out.params['p_0'].value, vary=False)
            parameters.add('p_1', value=out.params['p_1'].value, vary=False)
            if temperature is not None:
                parameters.add('p_2', value=out.params['p_2'].value, vary=False)
            else:
                parameters.add('p_2', value=0., vary=False)

            parameters.add('p_3', value=0.04)
            parameters.add('temperature_threshold', value=temperature_threshold, vary=False)
            if temperature is not None:
                parameters.add('temperature_sets', value=len(temperature), vary=False)
                for i in range(len(temperature)):
                    parameters.add('temperature_%d' % (i+1), value=temperature[i], vary=False)
            else:
                parameters.add('temperature_sets', value=0, vary=False)

            parameters.add('cell_size_sets', value=len(cell_size), vary=False)
            for i in range(len(cell_size)):
                parameters.add('cell_size_%d' % (i+1), value=cell_size[i], vary=False)

            out = minimize(self.objective, parameters, args=(raw_data_x, raw_data_y))
            return {'initial_resistance': initial_resistance, 'p_0': out.params['p_0'].value, 'p_1': out.params['p_1'].value, 'p_2': out.params['p_2'].value, 'p_3': out.params['p_3'].value, 'temperature_threshold': out.params['temperature_threshold'].value}
        elif self.operation_mode == OperationMode.sudden:
            if temperature is not None:
                return {'initial_resistance': initial_resistance, 'p_0': out.params['p_0'].value, 'p_1': out.params['p_1'].value, 'p_2': out.params['p_2'].value, 'temperature_threshold': out.params['temperature_threshold'].value, 'stable_resistance': stable_resistance}
            else:
                return {'initial_resistance': initial_resistance, 'p_0': out.params['p_0'].value, 'p_1': out.params['p_1'].value, 'p_2': 0., 'stable_resistance': stable_resistance}
