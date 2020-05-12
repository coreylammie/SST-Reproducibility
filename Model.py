import numpy as np
import pandas as pd
import math
import enum
from enum import Enum, auto

class OperationMode(Enum):
    sudden = auto()
    gradual = auto()


class Model():
    def __init__(self, operation_mode, tempurature_dependance=False, cell_size_dependance=False):
        self.operation_mode = operation_mode
        self.tempurature_dependance = tempurature_dependance


    def gradual_convergence(self, input, input_stable, pq_0, pq_1, pq_2, pq_3, r_on=True, cell_size=None):
        assert input is not None and len(input) > 0, 'input is invalid.'
        assert input.ndim == 1, 'input must be 1-dimensional.'
        output = np.zeros(input.shape)
        if cell_size is None:
            cell_size = 10

        output_stable = pq_1 * np.exp(pq_2 * cell_size)
        output[output <= output_stable] = 10 ** pq_0
        if r_on:
            output[output > output_stable] = 10 ** (pq_3 * cell_size * np.log10(input) + np.log10(10 ** pq_0) + pq_3 * np.log10(pq_1 * np.exp(pq_2 * cell_size)))
        else:
            output[output > output_stable] = 10 ** (pq_3 * cell_size * np.log10(input) + np.log10(10 ** pq_0) - pq_3 * np.log10(pq_1 * np.exp(pq_2 * cell_size)))

        return output
