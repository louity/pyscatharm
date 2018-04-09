"""Author: Louis Thiry, All rights reserved, 2018."""
from collections import defaultdict

import torch
from skcuda import cufft
import numpy as np
import pyfftw


def generate_sum_of_gaussians(centers, sigma, M, N, O, fourier=False):
    grid = np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3))
    if fourier:
        signals = np.zeros((centers.shape[0], M, N, O), dtype='complex64')
        grid[0] *= 2*np.pi / M
        grid[1] *= 2*np.pi / N
        grid[2] *= 2*np.pi / O
    else:
        signals = np.zeros((centers.shape[0], M, N, O), dtype='float32')

    for i_signal in range(centers.shape[0]):
        n_centers = centers[i_signal].shape[0]
        for i_center in range(n_centers):
            center = centers[i_signal, i_center].reshape((3, 1, 1, 1))
            if fourier:
                signals[i_signal] += np.exp(-1j * (grid*center).sum(0))
            else:
                signals[i_signal] += np.exp(-0.5 * ((grid - center)**2).sum(0) / sigma**2)
    if fourier:
        signals*= np.exp(-0.5 * ((grid*sigma)**2).sum(0))
        return signals
    else:
        signals /= (2 * np.pi)**1.5 * sigma**3
        return torch.from_numpy(signals)


def subsample(input, j):
    return input.unfold(3, 1, 2**j).unfold(2, 1, 2**j).unfold(1, 1, 2**j).contiguous()


def complex_modulus(input):
    modulus = input.new(input.size()).fill_(0)
    modulus[..., 0] += torch.sqrt((input**2).sum(-1))
    return modulus


def compute_integrals(input, integral_powers):
    """Computes integrals of the input to the given powers."""
    integrals = torch.zeros(input.size(0), len(integral_powers), 1)
    for i_q, q in enumerate(integral_powers):
        integrals[:, i_q, 0] = (input**q).view(input.size(0), -1).sum(1).cpu()
    return integrals


def get_3d_angles(cartesian_grid):
    """Given a cartisian grid, computes the spherical coord angles (theta, phi).
    Input args:
        cartesian_grid: 4D tensor of shape (3, M, N, O)
    Returns:
        polar, azimutal: 3D tensors of shape (M, N, O).
    """
    z, y, x = cartesian_grid
    azimuthal = np.arctan2(y, x)
    rxy = np.sqrt(x**2 + y**2)
    polar = np.arctan2(z, rxy) + np.pi / 2
    return polar, azimuthal


def double_factorial(l):
    return 1 if (l < 1) else np.prod(np.arange(l, 0, -2))


def getDtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


def iscomplex(input):
    return input.size(-1) == 2


def to_complex(input):
    output = input.new(input.size() + (2,)).fill_(0)
    output[..., 0] = input
    return output


class Fft3d(object):
    """This class builds a wrapper to 3D FFTW on CPU / cuFFT on nvidia GPU."""

    def __init__(self):
        self.fftw_cache = defaultdict(lambda: None)
        self.cufft_cache = defaultdict(lambda: None)

    def buildCufftCache(self, input, type):
        batch_size, M, N, O, _ = input.size()
        signal_dims = np.asarray([M, N, O], np.int32)
        batch = batch_size
        idist = M * N * O
        istride = 1
        ostride = istride
        odist = idist
        rank = 3
        print(rank, signal_dims.ctypes.data, signal_dims.ctypes.data, istride, idist, signal_dims.ctypes.data, ostride, odist, type, batch)
        plan = cufft.cufftPlanMany(rank, signal_dims.ctypes.data, signal_dims.ctypes.data,
                                   istride, idist, signal_dims.ctypes.data, ostride, odist, type, batch)
        self.cufft_cache[(input.size(), type, input.get_device())] = plan

    def buildFftwCache(self, input, inverse):
        direction = 'FFTW_BACKWARD' if inverse else 'FFTW_FORWARD'
        batch_size, M, N, O, _ = input.size()
        fftw_input_array = pyfftw.empty_aligned((batch_size, M, N, O), dtype='complex64')
        fftw_output_array = pyfftw.empty_aligned((batch_size, M, N, O), dtype='complex64')
        fftw_object = pyfftw.FFTW(fftw_input_array, fftw_output_array, axes=(1, 2, 3), direction=direction, threads=1)
        self.fftw_cache[(input.size(), inverse)] = (fftw_input_array, fftw_output_array, fftw_object)

    def __call__(self, input, inverse=False, normalized=False):
        if not isinstance(input, torch.cuda.FloatTensor):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                f = lambda x: np.stack((np.real(x), np.imag(x)), axis=len(x.shape))
                if(self.fftw_cache[(input.size(), inverse)] is None):
                    self.buildFftwCache(input, inverse)
                input_arr, output_arr, fftw_obj = self.fftw_cache[(input.size(), inverse)]

                input_arr[:] = input[..., 0].numpy() + 1.0j * input[..., 1].numpy()
                fftw_obj()

                return torch.from_numpy(f(output_arr).astype(input.numpy().dtype))

        assert input.is_contiguous()
        output = input.new(input.size())
        flag = cufft.CUFFT_INVERSE if inverse else cufft.CUFFT_FORWARD
        ffttype = cufft.CUFFT_C2C if isinstance(input, torch.cuda.FloatTensor) else cufft.CUFFT_Z2Z
        if (self.cufft_cache[(input.size(), ffttype, input.get_device())] is None):
            self.buildCufftCache(input, ffttype)
        cufft.cufftExecC2C(self.cufft_cache[(input.size(), ffttype, input.get_device())],
                           input.data_ptr(), output.data_ptr(), flag)
        if normalized:
            output /= input.size(1) * input.size(2) * input.size(3)
        return output


def cdgmm3d(A, B):
    """Pointwise multiplication of complex tensors."""
    A, B = A.contiguous(), B.contiguous()

    if A.size()[-4:] != B.size():
        raise RuntimeError('The tensors are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 4:
        raise RuntimeError('The second tensor must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    C = A.new(A.size())

    C[..., 0] = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    C[..., 1] = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]

    return C
