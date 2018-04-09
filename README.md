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

The first step is to install pytorch following instructions from
<http://pytorch.org>, then you can run `pip`:

```
pip install -r requirements.txt
python setup.py install
```

## Usage

Example:
```python
import numpy as np
from scatharm.scattering import SolidHarmonicScattering
from scatharm.utils import generate_sum_of_gaussians

centers = np.zeros((1, 1, 3))
sigma = 8.
M, N, O, J, L = 128, 128, 128, 0, 3
scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

x = generate_sum_of_gaussians(centers, sigma, M, N ,O)

# integral scattering coefficients
s_cpu = scat(x, order_2=False, method='integral', method_args={'integral_powers': [1]})
print('CPU', s_cpu)
s_gpu = scat(x_gpu, order_2=False, method='integral', method_args={'integral_powers': [1]})
print('GPU', s_gpu)

# standard scattering coefficients
s_cpu = scat(x, order_2=False, method='standard')
s_gpu = scat(x_gpu, order_2=False, method='standard')
print('GPU', s_gpu)
```

## Contribution

All contributions are welcome.


## Authors

Louis Thiry, based on code by Edouard Oyallon, Eugene Belilovsky and Sergey Zagoruyko
