"""Author: Louis Thiry, All rights reserved, 2018."""

import torch
import numpy as np

if torch.cuda.is_available():
    from .skcuda_utils import cublas
    CUDA = True
else:
    print('CUDA not available in torch, GPU version will not work')
    CUDA = False

if not torch.backends.mkl.is_available():
    print('MKL not available in torch, CPU version will not work')

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


def generate_large_weighted_sum_of_gaussians(positions, weights, M, N, O, fourier_gaussian):
    n_signals = positions.shape[0]
    signals = torch.FloatTensor(n_signals, M, N, O, 2).fill_(0)
    signals[..., 0] = generate_weighted_sum_of_diracs(positions, weights, M, N, O)

    return fft(cdgmm3d(fft(signals, inverse=False), fourier_gaussian), inverse=True)[..., 0]


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


def fft(input, inverse=False):
    """
        fft of a 3d signal
        Example
        -------
        x = torch.randn(128, 32, 32, 32, 2)
        x_fft = fft(x, inverse=True)
        Parameters
        ----------
        input : tensor
            complex input for the FFT
        inverse : bool
            True for computing the inverse FFT.
.
    """
    if not iscomplex(input):
        raise(TypeError('The input should be complex (e.g. last dimension is 2)'))
    if inverse:
        return torch.ifft(input, 3)
    return torch.fft(input, 3)


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
