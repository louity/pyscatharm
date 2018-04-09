""" This script will test the submodules used by the scattering module"""

import torch
import unittest
import numpy as np
from scatharm.scattering import SolidHarmonicScattering
from scatharm import utils as sl

def linfnorm(x,y):
    return torch.max(torch.abs(x-y))

class TestScattering(unittest.TestCase):
    def testFFT3dCentralFreqBatch(self):
        # Checked the 0 frequency for the 3D FFT
        for gpu in [False, True]:
            x = torch.FloatTensor(1, 32, 32, 32, 2).fill_(0)
            if gpu:
                x = x.cuda()

            a = x.sum()
            fft3d = sl.Fft3d()
            y = fft3d(x)
            c = y[:,0,0,0].sum()
            self.assertAlmostEqual(a, c, places=6)


    def testSolidHarmonicScattering(self):
        # Compare value to analytical formula in the case of a single Gaussian
        centers = np.zeros((1, 1, 3))
        sigma_gaussian = 3.
        sigma_0_wavelet = 3.
        M, N, O, J, L = 128, 128, 128, 1, 3
        x = sl.generate_sum_of_gaussians(centers, sigma_gaussian, M, N, O)
        scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma_0_wavelet)
        args = {'integral_powers': [1]}
        s = scat(x, order_2=False, method='integral', method_args=args)
        print(s.size())

        for j in range(J+1):
            sigma_wavelet = sigma_0_wavelet*2**j
            k = sigma_wavelet / np.sqrt(sigma_wavelet**2 + sigma_gaussian**2)
            for l in range(1, L+1):
                self.assertAlmostEqual(s[0, 0, j, l], k**l, places=4)


if __name__ == '__main__':
    unittest.main()
