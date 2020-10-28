# General Metal-Oxide ReRAM Device Endurance and Retention Model for Deep Learning Simulations

![](https://img.shields.io/badge/license-GPL-blue.svg)
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/)

Supplementary GitHub Repository containing code to reproduce results from Fig. 2 from *General Metal-Oxide ReRAM Device Endurance and Retention Model for Deep Learning Simulations*, which has been submitted to IEEE Electron Device Letters.

## Abstract
Memristive devices including Resistive Random Access Memory (ReRAM) cells are promising nanoscale low-power components projected to achieve significant improvement in power and speed of Deep Learning (DL) accelerators, if structured in crossbar architectures. However, these devices possess non-ideal endurance and retention properties, which should be modeled efficiently. In this letter, we propose a novel generalized Metal-Oxide ReRAM endurance and retention model for use in large-scale DL simulations. To the best of our knowledge, the proposed model is the first to unify retention-endurance modeling while taking into account time, energy, SET-RESET cycles, and device size. We compare the model to state-of-the-art, demonstrate its versatility by applying it to experimental data from fabricated devices, and use it for CIFAR-10 dataset classification using a large-scale Deep Memristive Neural Network (DMNN) to investigate performance degradation on account of device endurance and retention losses.

## Reproducability
We provide a set of Python and MATLAB scripts to reproduce results from Fig. 2. A Python interpreter (⩾3.6), the pip package manager, and MATLAB are required. All other Python dependencies can be installed using `pip install -r requirements.txt`.

1. Results from Figure 2A can be reproduced using [plot_2A.py](plot_2A.py).
2. Results from Figure 2B can be reproduced using [plot_2B.py](plot_2B.py).
3. Results from Figure 2C can be reproduced using [plot_2C.py](plot_2C.py).
4. Results from Figure 2D can be reproduced using [plot_2D.py](plot_2D.py).
5. Results from Figure 2E can be reproduced using [large_scale_simulations.ipynb](large_scale_simulations.ipynb) and [plot_2E.m](plot_2E.m).
6. Results from Figure 2F can be reproduced using [large_scale_simulations.ipynb](large_scale_simulations.ipynb) and [plot_2F.m](plot_2F.m).
7. Results from Figure 2G can be reproduced using [large_scale_simulations.ipynb](large_scale_simulations.ipynb) and [plot_2G.m](plot_2G.m).
8. Results from Figure 2H can be reproduced using [plot_2H.py](plot_2H.py).

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are avaliable at: https://www.gnu.org/licenses/gpl-3.0.en.html
