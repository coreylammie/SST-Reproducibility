# TiN/Hf(Al)O/Hf/TiN devices from Figure 2 (A)
import enum
from enum import Enum, auto
from memtorch.mn.Module import supported_module_parameters
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import memtorch
from memtorch.mn.Module import patch_model
from memtorch.map.Module import naive_tune
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
from memtorch.bh.crossbar.Crossbar import init_crossbar
import copy
from pprint import pprint
from mobilenetv2 import MobileNetV2
from scipy.interpolate import interp1d
import torchvision

def test(model, test_loader):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data.to(device))
        pred = output.data.max(1)[1]
        correct += pred.eq(target.to(device).data.view_as(pred)).cpu().sum()

    return 100. * float(correct) / float(len(test_loader.dataset))

def update_patched_model(patched_model, model):
    for i, (name, m) in enumerate(list(patched_model.named_modules())):
        if isinstance(m, memtorch.mn.Conv2d) or isinstance(m, memtorch.mn.Linear):
            pos_conductance_matrix, neg_conductance_matrix = naive_map(getattr(model, name).weight.data, r_on, r_off,scheme=memtorch.bh.Scheme.DoubleColumn)
            m.crossbars[0].write_conductance_matrix(pos_conductance_matrix, transistor=True, programming_routine=None)
            m.crossbars[1].write_conductance_matrix(neg_conductance_matrix, transistor=True, programming_routine=None)
            m.weight.data = getattr(model, name).weight.data

    return patched_model

scale_input = interp1d([1.3, 1.9], [0, 1])
def scale_p_0(p_0, p_1, v_stop, cell_size=10):
    scaled_input = scale_input(v_stop)
    x = 1.50
    y = p_0 * np.exp(p_1 * cell_size)
    k = np.log10(y) / (1 - (2 * scale_input(x) - 1) ** (2))
    return (10 ** (k * (1 - (2 * scaled_input - 1) ** (2)))) / (np.exp(p_1 * cell_size))

def gradual(initial_resistance, cycle_count, p_0, p_1, p_2, cell_size):
    threshold = p_0 * np.exp(p_1 * cell_size)
    return torch.pow(10, (p_2 * cell_size * np.log10(cycle_count) + torch.log10(initial_resistance) - p_2 * cell_size * np.log10(threshold)))

def model_gradual(layer, cycle_count, v_stop):
    cell_size = 20
    convergence_point_lrs = 1e4
    initial_resistance_lrs = 4400
    p_0_lrs = 10.0
    p_1_lrs = 0.6907755278982137
    p_2_lrs = 0.012171029136082496
    convergence_point_hrs = 1e4
    initial_resistance_hrs = 65000
    p_0_hrs = 10.0
    p_1_hrs = 0.6907755278982137
    p_2_hrs = -0.015855563129903612
    p_0_lrs = scale_p_0(p_0_lrs, p_1_lrs, v_stop, cell_size)
    p_0_hrs = scale_p_0(p_0_hrs, p_1_hrs, v_stop, cell_size)
    threshold_lrs = p_0_lrs * np.exp(p_1_lrs * cell_size)
    threshold_hrs = p_0_hrs * np.exp(p_1_hrs * cell_size)
    for i in range(len(layer.crossbars)):
        initial_resistance = 1 / layer.crossbars[i].conductance_matrix
        if initial_resistance[initial_resistance < convergence_point_lrs].nelement() > 0:
            if cycle_count > threshold_lrs:
                initial_resistance[initial_resistance < convergence_point_lrs] = gradual(initial_resistance[initial_resistance < convergence_point_lrs], cycle_count, p_0_lrs, p_1_lrs, p_2_lrs, cell_size)

        if initial_resistance[initial_resistance > convergence_point_hrs].nelement() > 0:
            if cycle_count > threshold_hrs:
                initial_resistance[initial_resistance > convergence_point_hrs] = gradual(initial_resistance[initial_resistance > convergence_point_hrs], cycle_count, p_0_hrs, p_1_hrs, p_2_hrs, cell_size)

        layer.crossbars[i].conductance_matrix = 1 / initial_resistance

    return layer

def model_degradation(model, cycle_count, v_stop):
    for i, (name, m) in enumerate(list(model.named_modules())):
        if type(m) in supported_module_parameters.values():
            setattr(model, name, model_gradual(m, cycle_count, v_stop))

    return model

device = torch.device('cuda')
batch_size = 256
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
reference_memristor = memtorch.bh.memristor.VTEAM
r_on = 4400
r_off = 65000
reference_memristor_params = {'time_series_resolution': 1e-10, 'r_off': r_off, 'r_on': r_on}
model = MobileNetV2().to(device)
model.load_state_dict(torch.load('trained_model.pt'), strict=False)
model.eval()
patched_model = patch_model(model,
                          memristor_model=reference_memristor,
                          memristor_model_params=reference_memristor_params,
                          module_parameters_to_patch=[torch.nn.Linear, torch.nn.Conv2d],
                          mapping_routine=naive_map,
                          transistor=True,
                          programming_routine=None,
                          scheme=memtorch.bh.Scheme.DoubleColumn,
                          tile_shape=(128, 128),
                          max_input_voltage=0.3,
                          ADC_resolution=8,
                          ADC_overflow_rate=0.,
                          quant_method='linear')
del model
patched_model.tune_()
times_to_reprogram = 10 ** np.arange(1, 4, dtype=np.float64)
v_stop_values = np.linspace(1.3, 1.9, 10, endpoint=True)
df = pd.DataFrame(columns=['times_reprogramed', 'v_stop', 'test_set_accuracy'])
for time_to_reprogram in times_to_reprogram:
    cycle_count = time_to_reprogram
    for v_stop in v_stop_values:
        print('time_to_reprogram: %f, v_stop: %f' % (time_to_reprogram, v_stop))
        patched_model_copy = copy.deepcopy(patched_model)
        patched_model_copy = model_degradation(patched_model_copy, cycle_count, v_stop)
        accuracy = test(patched_model_copy, test_loader)
        del patched_model_copy
        df = df.append({'times_reprogramed': time_to_reprogram, 'v_stop': v_stop, 'test_set_accuracy': accuracy}, ignore_index=True)
        df.to_csv('6E.csv', index=False)
