"""Author: Louis Thiry, All rights reserved, 2018."""

__all__ = ['SolidHarmonicScattering']

import torch
from .utils import cdgmm3d, Fft3d, compute_integrals, subsample, complex_modulus, to_complex
from .filters_bank import solid_harmonic_filters_bank, gaussian_filters_bank


class SolidHarmonicScattering(object):
    """Scattering module.

    Runs solid scattering on an input 3D image

    Input args:
        M, N, O: input 3D image size
        J: number of scales
        L: number of l values
    """
    def __init__(self, M, N, O, J, L, sigma_0):
        super(SolidHarmonicScattering, self).__init__()
        self.M, self.N, self.O, self.J, self.L, self.sigma_0 = M, N, O, J, L, sigma_0
        self.filters = solid_harmonic_filters_bank(self.M, self.N, self.O, self.J, self.L, sigma_0)
        self.gaussian_filters = gaussian_filters_bank(self.M, self.N, self.O, self.J+1, sigma_0)
        self.fft = Fft3d()

    def _fft_convolve(self, input, filter):
        return self.fft(cdgmm3d(self.fft(input, inverse=False), filter), inverse=True, normalized=True)

    def _low_pass_filter(self, input, j):
        cuda = isinstance(input, torch.cuda.FloatTensor)
        low_pass = self.gaussian_filters[j].type(torch.cuda.FloatTensor) if cuda else self.gaussian_filters[j]
        return self._fft_convolve(input, low_pass)

    def _compute_standard_scattering_coefs(self, input):
        convolved_input = self._low_pass_filter(input, self.J)
        return subsample(convolved_input, self.J).view(input.size(0), -1, 1)

    def _compute_local_scattering_coefs(self, input, points, j):
        local_coefs = torch.zeros(input.size(0), points.size(1), 1)
        convolved_input = self._low_pass_filter(input, j+1)
        for i in range(input.size(0)):
            for j in range(points[i].size(0)):
                x, y, z = points[i, j, 0], points[i, j, 1], points[i, j, 2]
                local_coefs[i, j, 0] = convolved_input[i, int(x), int(y), int(z), 0]
        return local_coefs

    def _compute_scattering_coefs(self, input, method, args, j):
        methods = ['standard', 'local', 'integral']
        if (not method in methods):
            raise(ValueError('method must be in {}'.format(methods)))
        if method == 'integral':
            return compute_integrals(input[..., 0], args['integral_powers'])
        elif method == 'local':
            return self._compute_local_scattering_coefs(input, args['points'], j)
        elif method == 'standard':
            return self._compute_standard_scattering_coefs(input)

    def _rotation_covariant_convolution_and_modulus(self, input, l, j):
        cuda = isinstance(input, torch.cuda.FloatTensor)
        filters_l_j = self.filters[l][j].type(torch.cuda.FloatTensor) if cuda else self.filters[l][j]
        convolution_modulus = input.new(input.size()).fill_(0)
        for m in range(filters_l_j.size(0)):
            convolution_modulus[..., 0] += (self._fft_convolve(input, filters_l_j[m])**2).sum(-1)
        return torch.sqrt(convolution_modulus)

    def _convolution_and_modulus(self, input, l, j, m=0):
        cuda = isinstance(input, torch.cuda.FloatTensor)
        filters_l_m_j = self.filters[l][j][m].type(torch.cuda.FloatTensor) if cuda else self.filters[l][j][m]
        return complex_modulus(self._fft_convolve(input, filters_l_m_j))

    def forward(self, input, order_2=True, rotation_covariant=True, method='standard', method_args=None):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            input = input.contiguous()

        if((input.size(-1)!=self.O or input.size(-2)!=self.N or input.size(-3)!=self.M)):
            raise (RuntimeError('Tensor must be of spatial size (%i,%i,%i)!'%(self.M,self.N,self.O)))

        if (input.dim() != 4):
            raise (RuntimeError('Input tensor must be 4D'))

        convolution_and_modulus = \
            self._rotation_covariant_convolution_and_modulus if rotation_covariant else self._convolution_and_modulus
        compute_scattering_coefs = self._compute_scattering_coefs

        s_order_1 = []
        s_order_2 = []
        _input = to_complex(input)

        for l in range(self.L+1):
            s_order_1_l, s_order_2_l = [], []
            for j_1 in range(self.J+1):
                conv_modulus = convolution_and_modulus(_input, l, j_1)
                s_order_1_l.append(compute_scattering_coefs(conv_modulus, method, method_args, j_1))
                if not order_2:
                    continue
                for j_2 in range(j_1+1, self.J+1):
                    conv_modulus_2 = convolution_and_modulus(conv_modulus, l, j_2)
                    s_order_2_l.append(compute_scattering_coefs(conv_modulus_2, method, method_args, j_2))
            s_order_1.append(torch.cat(s_order_1_l, -1))
            if order_2:
                s_order_2.append(torch.cat(s_order_2_l, -1))

        if order_2:
            return torch.stack(s_order_1, dim=-1), torch.stack(s_order_2, dim=-1)
        return torch.stack(s_order_1, dim=-1)

    def __call__(self, input, order_2=False, method='standard', method_args=None):
        return self.forward(input, order_2=order_2, method=method, method_args=method_args)
