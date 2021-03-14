# Ti/HfAlO/TiN devices from Figure 3 (B)
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

def gradual(initial_resistance, time, p_0, p_2, p_3, tempurature_constant):
    threshold = p_0 * np.exp(p_2 * tempurature_constant)
    return torch.pow(10, (p_3 * (p_2 * tempurature_constant) * np.log10(time) + torch.log10(initial_resistance) - p_3 * (p_2 * tempurature_constant) * np.log10(threshold)))

def model_gradual(layer, time, tempurature):
    cell_size = 10
    convergence_point_lrs = 5e6
    threshold_lrs = 298
    initial_resistance_lrs = 4250
    p_0_lrs = 4.764e-17
    p_2_lrs = 76.47
    p_3_lrs = 0.014
    tempurature_threshold_lrs = 298
    tempurature_constant_lrs = np.min(tempurature_threshold_lrs / tempurature, 1)
    threshold_lrs = p_0_lrs * np.exp(p_2_lrs * tempurature_constant_lrs)
    for i in range(len(layer.crossbars)):
        initial_resistance = 1 / layer.crossbars[i].conductance_matrix
        if initial_resistance[initial_resistance < convergence_point_lrs].nelement() > 0:
            if time > threshold_lrs:
                initial_resistance[initial_resistance < convergence_point_lrs] = gradual(initial_resistance[initial_resistance < convergence_point_lrs], time, p_0_lrs, p_2_lrs, p_3_lrs, tempurature_constant_lrs)

        layer.crossbars[i].conductance_matrix = 1 / initial_resistance

    return layer

def model_degradation(model, time, tempurature):
    for i, (name, m) in enumerate(list(model.named_modules())):
            if type(m) in supported_module_parameters.values():
                setattr(model, name, model_gradual(m, time, tempurature))

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
r_on = 4250
r_off = 5e6
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
times = 10 ** np.arange(1, 10, dtype=np.float64)
tempuratures = np.linspace(473, 573, 10, endpoint=True)
df = pd.DataFrame(columns=['time', 'tempurature', 'test_set_accuracy'])
for time_ in times:
    for tempurature in tempuratures:
        print('time: %f, tempurature: %f' % (time_, tempurature))
        patched_model_copy = copy.deepcopy(patched_model)
        patched_model_copy = model_degradation(patched_model_copy, time_, tempurature)
        accuracy = test(patched_model_copy, test_loader)
        del patched_model_copy
        df = df.append({'time': time_, 'tempurature': tempurature, 'test_set_accuracy': accuracy}, ignore_index=True)
        df.to_csv('6G.csv', index=False)
