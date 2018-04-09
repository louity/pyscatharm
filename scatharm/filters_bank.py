"""
Authors: Eugene Belilovsky, Edouard Oyallon and Sergey Zagoruyko
All rights reserved, 2017.
"""

__all__ = ['solid_harmonic_filters_bank']

import torch
import numpy as np
from scipy.special import sph_harm, factorial
from .utils import get_3d_angles, double_factorial


def solid_harmonic_filters_bank(M, N, O, J, L, sigma_0, fourier=True):
    filters = []
    for l in range(L+1):
        filters_l = np.zeros((J+1, 2*l+1, M, N, O, 2), dtype='float32')
        for j in range(J+1):
            sigma = sigma_0 * 2**j
            solid_harm = solid_harmonic_3d(M, N, O, sigma, l, fourier=fourier)
            filters_l[j, :, :, :, :, 0] = solid_harm.real
            filters_l[j, :, :, :, :, 1] = solid_harm.imag
        filters.append(torch.from_numpy(filters_l))
    return filters


def gaussian_filters_bank(M, N, O, J, sigma_0, fourier=True):
    gaussians = torch.zeros(J+1, M, N, O, 2)
    for j in range(J+1):
        sigma = sigma_0 * 2**j
        gaussian = gaussian_3d(M, N, O, sigma, fourier=fourier)
        gaussians[j, :, :, :, 0] = torch.from_numpy(gaussian)
    return gaussians


def gaussian_3d(M, N, O, sigma, fourier=True):
    """Computes gaussian in Fourier or signal space."""
    grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32')
    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        sigma = 1. / sigma

    gaussian = np.exp(-0.5 * (grid**2).sum(0) / sigma**2)
    if not fourier:
        gaussian /= (2 * np.pi)**1.5 * sigma**3

    return np.fft.ifftshift(gaussian)


def solid_harmonic_3d(M, N, O, sigma, l, fourier=True):
    """Computes solid harmonic wavelets in Fourier or signal space.

    Input args:
        M, N, O: integers, shape of the grid
        sigma: float, width of the wavelets
        l: integer, degree of the harmonic
        fourier: boolean, compute wavelet in fourier space
                 or in signal space

    Returns:
        solid_harm: 4D tensors of shape (2l+1, M, N, O). The
                    tensor is ifftshifted such that the point 0 in
                    signal space or in Fourier space is at
                    [m, 0, 0, 0] for m = 0 ... 2*l+1
    """
    solid_harm = np.zeros((2*l+1, M, N, O), np.complex64)
    grid = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32')

    if fourier:
        grid[0] *= 2 * np.pi / M
        grid[1] *= 2 * np.pi / N
        grid[2] *= 2 * np.pi / O
        sigma = 1. / sigma

    r_square = (grid**2).sum(0)

    if l == 0:
        gaussian = np.exp(-0.5 * r_square / sigma**2).reshape((1, M, N, O)).astype('complex64')
        if fourier:
            return gaussian
        return gaussian / ((2 *np.pi)**1.5 * sigma**3)

    polynomial_gaussian = r_square**(0.5*l) / sigma**l * np.exp(-0.5 * r_square / sigma**2)

    polar, azimuthal = get_3d_angles(grid)

    for i_m, m in enumerate(range(-l, l+1)):
        solid_harm[i_m] = sph_harm(m, l, azimuthal, polar) * polynomial_gaussian

    if l % 2 == 0:
        norm_factor = 1. / (2 * np.pi * np.sqrt(l+0.5) * double_factorial(l+1))
    else :
        norm_factor = 1. / (2**(0.5*(l+3)) * np.sqrt(np.pi*(2*l+1)) * factorial((l+1)/2))

    if fourier:
        norm_factor *= (2 * np.pi)**1.5 * (-1j)**l
    else:
        norm_factor /= sigma**3

    solid_harm *= norm_factor

    return np.fft.ifftshift(solid_harm)
