# A General Metal Oxide ReRAM Device Endurance and Retention Model for Large-scale Simulations

![](https://img.shields.io/badge/license-GPL-blue.svg)

Supplementary GitHub Repository containing code to reproduce results from Fig. 2 and Fig. 3[A-B] from *A General Metal Oxide ReRAM Device Endurance and Retention Model for Large-scale Simulations*, which has been submitted to IEEE Electron Device Letters.

## Abstract
Metal-Oxide Resistive Random Access Memory (ReRAM) devices are a class of memristors, that when arranged in crossbar configurations, can perform in-memory computing, which has shown great promise to augment the performance of Machine Learning (ML) and neuromorphic architectures. However, these devices face concerns of device failure and non-idealities, which greatly inhibits their performance and usability. In this letter, we propose a novel computationally efficient generalized Metal Oxide ReRAM endurance and retention model for use in large-scale simulations. We compare it to related works, demonstrate its versatility by fitting it to experimental data from various fabricated devices, and use it within large-scale simulations with a Memristive Deep Neural Network (MDNN) determining performance using the CIFAR-10 dataset. 

## Reproducability
We provide a set of Python scripts to reproduce results from Fig. 2 and Fig. 3[A-B]. A Python interpreter (⩾3.6) and the pip package manager are required. All other dependencies can be installed using `pip install -r requirements.txt`.


1. Results from Figure 2A can be reproduced using [plot_2A.py](plot_2A.py).
2. Results from Figure 2B can be reproduced using [plot_2B.py](plot_2B.py).
3. Results from Figure 3A can be reproduced using [plot_3A.py](plot_3A.py).
4. Results from Figure 3B can be reproduced using [plot_3B.py](plot_3B.py).

## License
All code is licensed under the GNU General Public License v3.0. Details pertaining to this are avaliable at: https://www.gnu.org/licenses/gpl-3.0.en.html
