# Empirical Metal-Oxide RRAM Device Endurance and Retention Model for Deep Learning Simulations

![](https://img.shields.io/badge/license-GPL-blue.svg)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)

Supplementary GitHub Repository containing code to reproduce results from *Empirical Metal-Oxide RRAM Device Endurance and Retention Model for Deep Learning Simulations*, which has been submitted to IOP Semiconductor Science and Technology.

## Abstract
Memristive devices including Resistive Random Access Memory (RRAM) cells are promising nanoscale low-power components projected to achieve significant improvement in power and speed of Deep Learning (DL) accelerators, if structured in crossbar architectures. However, these devices possess non-ideal endurance and retention properties, which should be modeled efficiently. In this paper, we propose a novel generalized Metal-Oxide RRAM endurance and retention model for use in large-scale DL simulations. To the best of our knowledge, the proposed model is the first to unify retention-endurance modeling while taking into account time, energy, SET-RESET cycles, and device size. We compare the model to state-of-the-art, demonstrate its versatility by applying it to experimental data from fabricated devices. Furthermore, we use the model for CIFAR-10 dataset classification using a large-scale Deep Memristive Neural Network (DMNN) implementing the MobileNetV2 architecture. Our results show that, even when ignoring other device non-idealities, retention and endurance losses significantly affect the performance of DL networks. Our proposed model and its DL simulations are made publicly available.

## Reproducability
We provide a set of Python and MATLAB scripts to reproduce results from Figures 3-5. A Python interpreter (⩾3.6), the pip package manager, and MATLAB are required. All other Python dependencies can be installed using `pip install -r requirements.txt`.

1. Results from Figure 2A can be reproduced using [plot_2A.py](plot_2A.py).
2. Results from Figure 2B can be reproduced using [plot_2B.py](plot_2B.py).
3. Results from Figure 2C can be reproduced using [plot_2C.py](plot_2C.py).
4. Results from Figure 3 can be reproduced using [plot_3.py](plot_3.py).
5. Results from Figure 4 can be reproduced using [plot_4.py](plot_4.py).
6. Results from Figure 5A can be reproduced using [large_scale_simulations.ipynb](large_scale_simulations.ipynb), [mobilenetv2.py](mobilenetv2.py), and [plot_5A.m](plot_5A.m).
7. Results from Figure 5B can be reproduced using [large_scale_simulations.ipynb](large_scale_simulations.ipynb), [mobilenetv2.py](mobilenetv2.py), and [plot_5B.m](plot_5B.m).
8. Results from Figure 5C can be reproduced using [large_scale_simulations.ipynb](large_scale_simulations.ipynb), [mobilenetv2.py](mobilenetv2.py), and [plot_5C.m](plot_5C.m).

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are avaliable at: https://www.gnu.org/licenses/gpl-3.0.en.html
