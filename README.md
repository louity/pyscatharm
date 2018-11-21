PyScatHarm
==========

PyTorch implementation of Solid Harmonic Scattering for 3D signals

A scattering network is a Convolutional Network with filters predefined to be wavelets that are not learned.
Intially designed in vision task such as classification of images using 2D gabor wavelets, this 3D version using solid harmonic wavelets gave close to state of the art results in molecular properties regression from the dataset QM9.

The software uses NumPy, Scipy and PyTorch.
Since Pytorch FFT use Intel MKL FFT on CPU, CPU version only works on Intel CPU.
Since Pytorch FFT use CUDA on GPU, GPU version only works on Nvidia GPU.


## Installation

The software was tested on Linux with anaconda Python 3.6 and
various GPUs, including Titan X, 1080s, 980s, K20s, and Titan X Pascal.

For the GPU version, you need to install [CUDA](https://developer.nvidia.com/cuda-downloads).

For the CPU version you need to install FFTW3.

Install [PyTorch](https://pytorch.org/get-started/locally/).
Then install the requirements:
```
pip install -r requirements.txt
```
and you can finally install Pyscatharm:
```
python setup.py install
```

## Usage

Example:
```python
import numpy as np
import torch
from scatharm.scattering import SolidHarmonicScattering
from scatharm.utils import generate_weighted_sum_of_gaussians

sigma = 4.
M, N, O, j_values, L = 128, 128, 128, [0], 3
centers = torch.FloatTensor(1, 1, 3).fill_(0)
weights = torch.FloatTensor(1, 1).fill_(1)
grid = torch.from_numpy(
    np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))

scat = SolidHarmonicScattering(M=M, N=N, O=O, j_values=j_values, L=L, sigma_0=sigma)

x_cpu = generate_weighted_sum_of_gaussians(grid, centers, weights, sigma)
s_cpu = scat(x_cpu, order_2=False, method='integral', method_args={'integral_powers': [1]})
print('CPU integral', s_cpu)

if torch.cuda.is_available():
	x_gpu = x_cpu.cuda()
	s_gpu = scat(x_gpu, order_2=False, method='integral', method_args={'integral_powers': [1]})
	print('GPU integral', s_gpu)
```

## Example scripts

The script `examples/qm.py` computes the Solid Harmonic scattering coefficients of the molecules of the QM7 or QM9 databases and runs linear regression to predict the atomization energies. You need to install [CheML](https://github.com/CheML/CheML)  and scikit-learn  (`pip install scikit-learn`) to run this example script.

## Contribution

All contributions are welcome.


## Authors

Louis Thiry inspired by code from Edouard Oyallon, Eugene Belilovsky and Sergey Zagoruyko
