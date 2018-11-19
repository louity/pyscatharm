"""Author: Louis Thiry, All rights reserved, 2018."""
from collections import defaultdict

import torch
import numpy as np
import pyfftw

if torch.cuda.is_available():
    from .skcuda_utils import cufft
    from .skcuda_utils import cublas
    CUDA = True
else:
    print('CUDA not available in torch, GPU version will not work')
    CUDA = False

def is_cuda_float_tensor(tensor):
    if not CUDA:
        return False
    return isinstance(tensor, torch.cuda.FloatTensor)


def generate_weighted_sum_of_diracs(positions, weights, M, N, O, sigma_dirac=0.4):
    n_signals = positions.shape[0]
    signals = torch.FloatTensor(n_signals, M, N, O).fill_(0)
    d_s = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    values = torch.FloatTensor(8)

    for i_signal in range(n_signals):
        n_positions = positions[i_signal].shape[0]
        for i_position in range(n_positions):
            position = positions[i_signal, i_position]
            i, j, k  = torch.floor(position).type(torch.IntTensor)
            for i_d, (d_i, d_j, d_k) in enumerate(d_s):
                values[i_d] = np.exp(-0.5 * (
                    (position[0] - (i+d_i))**2 + (position[1] - (j+d_j))**2 + (position[2] - (k+d_k))**2) / sigma_dirac**2)
            values *= weights[i_signal, i_position] / values.sum()
            for i_d, (d_i, d_j, d_k) in enumerate(d_s):
                i_, j_, k_ = (i+d_i) % M, (j+d_j) % N, (k+d_k) % O
                signals[i_signal, i_, j_, k_] += values[i_d]

    return signals


def generate_large_weighted_sum_of_gaussians(positions, weights, M, N, O, fourier_gaussian, fft=None):
    n_signals = positions.shape[0]
    signals = torch.FloatTensor(n_signals, M, N, O, 2).fill_(0)
    signals[..., 0] = generate_weighted_sum_of_diracs(positions, weights, M, N, O)

    if fft is None:
        fft = Fft3d()
    return fft(cdgmm3d(fft(signals, inverse=False), fourier_gaussian), inverse=True, normalized=True)[..., 0]


def generate_weighted_sum_of_gaussians_in_fourier_space(grid, positions, weights, sigma, cuda=False):
    _, M, N, O = grid.size()
    if cuda:
        signals = torch.cuda.FloatTensor(positions.size(0), M, N, O, 2).fill_(0)
    else:
        signals = torch.FloatTensor(positions.size(0), M, N, O, 2).fill_(0)

    gaussian = torch.exp(-0.5 * sigma**2 * (grid**2).sum(0))

    for i_signal in range(positions.size(0)):
        n_points = positions[i_signal].size(0)
        for i_point in range(n_points):
            if weights[i_signal, i_point] == 0:
                break
            weight = weights[i_signal, i_point]
            center = positions[i_signal, i_point]
            grid_dot_center = -(grid[0]*center[0] + grid[1]*center[1] + grid[2]*center[2])
            signals[i_signal, ..., 0] += weight * gaussian * torch.cos(grid_dot_center)
            signals[i_signal, ..., 1] += weight * gaussian * torch.sin(grid_dot_center)

    return signals


def generate_weighted_sum_of_gaussians(grid, positions, weights, sigma, cuda=False):
    _, M, N, O = grid.size()
    if cuda:
        signals = torch.cuda.FloatTensor(positions.size(0), M, N, O).fill_(0)
    else:
        signals = torch.FloatTensor(positions.size(0), M, N, O).fill_(0)

    for i_signal in range(positions.size(0)):
        n_points = positions[i_signal].size(0)
        for i_point in range(n_points):
            if weights[i_signal, i_point] == 0:
                break
            weight = weights[i_signal, i_point]
            center = positions[i_signal, i_point]
            signals[i_signal] += weight * torch.exp(
                -0.5 * ((grid[0]-center[0])**2 + (grid[1]-center[1])**2 + (grid[2]-center[2])**2) / sigma**2)
    return signals / ((2 * np.pi)**1.5 * sigma**3)


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


def iscomplex(input):
    return input.size(-1) == 2


def to_complex(input):
    output = input.new(input.size() + (2,)).fill_(0)
    output[..., 0] = input
    return output


class Fft3d(object):
    """This class builds a wrapper to 3D FFTW on CPU / cuFFT on nvidia GPU."""

    def __init__(self, n_fftw_threads=8):
        self.n_fftw_threads = n_fftw_threads
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
        fftw_object = pyfftw.FFTW(fftw_input_array, fftw_output_array, axes=(1, 2, 3), direction=direction,
                                  threads=self.n_fftw_threads)
        self.fftw_cache[(input.size(), inverse)] = (fftw_input_array, fftw_output_array, fftw_object)

    def __call__(self, input, inverse=False, normalized=False):
        if not is_cuda_float_tensor(input):
            if not isinstance(input, (torch.FloatTensor, torch.DoubleTensor)):
                raise(TypeError('The input should be a torch.cuda.FloatTensor, \
                                torch.FloatTensor or a torch.DoubleTensor'))
            else:
                f = lambda x: np.stack([x.real, x.imag], axis=len(x.shape))
                if(self.fftw_cache[(input.size(), inverse)] is None):
                    self.buildFftwCache(input, inverse)
                input_arr, output_arr, fftw_obj = self.fftw_cache[(input.size(), inverse)]

                input_arr.real[:] = input[..., 0]
                input_arr.imag[:] = input[..., 1]
                fftw_obj()

                return torch.from_numpy(f(output_arr))

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


def cdgmm3d(A, B, inplace=False, conjugate=False, use_cublas=True):
    """
        Complex pointwise multiplication between (batched) tensor A and tensor B.
        Parameters
        ----------
        A : tensor
            input tensor with size (batch_size, M, N, O, 2)
        B : tensor
            B is a complex tensor of size (M, N, O, 2)
        inplace : boolean, optional
            if set to True, all the operations are performed inplace
        conjugate : boolean, optional
            if set to True, B in complex conjugated
        conjugate : boolean, optional
            if set to True, use cublas Cdgmm
        Returns
        -------
        C : tensor
            output tensor of size (batch_size, M, N, O, 2) such that:
            C[b, m, n, o, :] = A[b, m, n, o, :] * B[m, n, o,:]
    """
    A, B = A.contiguous(), B.contiguous()

    if A.size()[-4:] != B.size():
        raise RuntimeError('The tensors are not compatible for multiplication!')

    if not iscomplex(A) or not iscomplex(B):
        raise TypeError('The input, filter and output should be complex')

    if B.ndimension() != 4:
        raise RuntimeError('The second tensor must be simply a complex array!')

    if type(A) is not type(B):
        raise RuntimeError('A and B should be same type!')

    if use_cublas and is_cuda_float_tensor(A) and is_cuda_float_tensor(B):
        C = A.new(A.size()) if not inplace else A
        m, n = B.nelement() // 2, A.nelement() // B.nelement()
        lda = m
        ldc = m
        incx = 1
        handle = torch.cuda.current_blas_handle()
        stream = torch.cuda.current_stream()._as_parameter_
        cublas.cublasSetStream(handle, stream)
        cublas.cublasCdgmm(handle, 'l', m, n, A.data_ptr(), lda, B.data_ptr(), incx, C.data_ptr(), ldc)
        return C

    C = A.new(A.size())

    if conjugate:
        C[..., 0] = A[..., 0] * B[..., 0] + A[..., 1] * B[..., 1]
        C[..., 1] = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]
        return C

    C[..., 0] = A[..., 0] * B[..., 0] - A[..., 1] * B[..., 1]
    C[..., 1] = A[..., 0] * B[..., 1] + A[..., 1] * B[..., 0]

    return C if not inplace else A.copy_(C)
