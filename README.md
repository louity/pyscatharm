PyScatHarm
==========

PyTorch implementation of Solid Harmonic Scattering for 3D signals

A scattering network is a Convolutional Network with filters predefined to be wavelets that are not learned.
Intially designed in vision task such as classification of images using 2D gabor wavelets, this 3D version using solid harmonic wavelets gave close to state of the art results in molecular properties regression from the dataset QM9.

The software uses NumPy with PyTorch + PyFFTW on CPU, and PyTorch + CuFFT on GPU.

This code was very largely inspired by [*PyScatWave*](https://github.com/edouardoyallon/pyscatwave) by E. Oyallon, E. Belilovsky, S. Zagoruyko, [*Scaling the Scattering Transform: Deep Hybrid Networks*](https://arxiv.org/abs/1703.08961)

## Benchmarks
TODO

## Installation

The software was tested on Linux with anaconda Python 3.6 and
various GPUs, including Titan X, 1080s, 980s, K20s, and Titan X Pascal.

For the GPU version, you need to install [CUDA](https://developer.nvidia.com/cuda-downloads).

For the CPU version you need to install FFTW3.
On Ubuntu :
```sudo apt-get install libfftw3-dev libfftw3-doc```

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

The script `examples/qm.py` computes the Solid Harmonic scattering coefficients of the molecules of the QM7 or QM9 databases. You need to install [CheML](https://github.com/CheML/CheML) to run this example script.

## Contribution

All contributions are welcome.


## Authors

Louis Thiry, based on code by Edouard Oyallon, Eugene Belilovsky and Sergey Zagoruyko
