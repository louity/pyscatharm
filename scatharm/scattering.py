"""Author: Louis Thiry, All rights reserved, 2018."""

__all__ = ['SolidHarmonicScattering']

import torch
from .utils import is_cuda_float_tensor, cdgmm3d, compute_integrals, subsample, complex_modulus, to_complex, fft
from .filters_bank import solid_harmonic_filters_bank, gaussian_filters_bank


class SolidHarmonicScattering(object):
    """Scattering module.

    Runs solid scattering on an input 3D image

    Input args:
        M, N, O: input 3D image size
        j_values: values for the scale parameter j
        L: number of l values """
    def __init__(self, M, N, O, j_values, L, sigma_0, fourier_grid=None, use_torch_fft=False):
        super(SolidHarmonicScattering, self).__init__()
        self.M, self.N, self.O, self.j_values, self.L, self.sigma_0 = M, N, O, j_values, L, sigma_0
        if fourier_grid is not None:
            self.filters = solid_harmonic_filters_bank(self.M, self.N, self.O, self.j_values, self.L, sigma_0, fourier_grid=fourier_grid)
            #TODO: add gaussian filters computations
        else:
            self.filters = solid_harmonic_filters_bank(self.M, self.N, self.O, self.j_values, self.L, sigma_0)
            self.gaussian_filters = gaussian_filters_bank(self.M, self.N, self.O, self.j_values, sigma_0)
        self.fft = fft

    def _fft_convolve(self, input, filter, fourier_input=False):
        f_input = input if fourier_input else self.fft(input, inverse=False)
        return self.fft(cdgmm3d(f_input, filter), inverse=True)

    def _low_pass_filter(self, input, i_j):
        cuda = is_cuda_float_tensor(input)
        low_pass = self.gaussian_filters[i_j].type(torch.cuda.FloatTensor) if cuda else self.gaussian_filters[i_j]
        return self._fft_convolve(input, low_pass)

    def _compute_standard_scattering_coefs(self, input):
        convolved_input = self._low_pass_filter(input, self.j_values[-1])
        return subsample(convolved_input, self.j_values[-1]).view(input.size(0), -1, 1)

    def _compute_local_scattering_coefs(self, input, points, i_j):
        local_coefs = torch.zeros(input.size(0), points.size(1), 1)
        convolved_input = self._low_pass_filter(input, i_j)
        for i in range(input.size(0)):
            for i_point in range(points[i].size(0)):
                x, y, z = points[i, i_point]
                local_coefs[i, i_point, 0] = convolved_input[i, int(x), int(y), int(z), 0]
        return local_coefs

    def _compute_scattering_coefs(self, input, method, args):
        methods = ['standard', 'local', 'integral']
        if (not method in methods):
            raise(ValueError('method must be in {}'.format(methods)))
        if method == 'integral':
            return compute_integrals(input[..., 0], args['integral_powers'])
        elif method == 'local':
            return self._compute_local_scattering_coefs(input, args['points'], args['j'])
        elif method == 'standard':
            return self._compute_standard_scattering_coefs(input)

    def _rotation_covariant_convolution_and_modulus(self, input, l, i_j, fourier_input=False):
        cuda = is_cuda_float_tensor(input)
        filters_l_j = self.filters[l][i_j].type(torch.cuda.FloatTensor) if cuda else self.filters[l][i_j]
        convolution_modulus = input.new(input.size()).fill_(0)
        for m in range(filters_l_j.size(0)):
            convolution_modulus[..., 0] += (self._fft_convolve(input, filters_l_j[m], fourier_input=fourier_input)**2).sum(-1)
        return torch.sqrt(convolution_modulus)


    def _convolution_and_modulus(self, input, l, i_j, m=0):
        cuda = is_cuda_float_tensor(input)
        filters_l_m_j = self.filters[l][i_j][m].type(torch.cuda.FloatTensor) if cuda else self.filters[l][i_j][m]
        return complex_modulus(self._fft_convolve(input, filters_l_m_j))


    def _check_input(self, input, fourier_input=False):
        if not torch.is_tensor(input):
            raise(TypeError('The input should be a torch.cuda.FloatTensor, a torch.FloatTensor or a torch.DoubleTensor'))

        if (not input.is_contiguous()):
            input = input.contiguous()

        if fourier_input:
            if((input.size(-1)!=2 or input.size(-2)!=self.O or input.size(-3)!=self.N or input.size(-4)!=self.M)):
                raise (RuntimeError('Fourier input tensor must be of spatial size (%i,%i,%i,%i)!'%(self.M,self.N,self.O,2)))

            if (input.dim() != 5):
                raise (RuntimeError('Fourier input tensor must be 5D'))
        else:
            if((input.size(-1)!=self.O or input.size(-2)!=self.N or input.size(-3)!=self.M)):
                raise (RuntimeError('Tensor must be of spatial size (%i,%i,%i)!'%(self.M,self.N,self.O)))

            if (input.dim() != 4):
                raise (RuntimeError('Input tensor must be 4D'))


    def compute_heatmap(self, input, integral_powers, order_0_weights, order_1_weights,
            order_2_weights=None, order_2=True, order_2_integer_j=False):
        # self._check_input(input)

        assert order_0_weights.shape == (len(integral_powers),), 'wrong shape for order_0_weights'
        assert order_1_weights.shape == (len(integral_powers), len(self.j_values), self.L+1,), 'wrong shape for order_1_weights'

        _input = to_complex(input)

        convolution_and_modulus = self._rotation_covariant_convolution_and_modulus

        heatmap = input.new(input.size()).fill_(0)

        for i_q, q in enumerate(integral_powers):
            heatmap += order_0_weights[i_q] * input**q


        for l in range(self.L+1):
            i_j2_weights = 0
            for i_j1 in range(len(self.j_values)):
                conv_modulus = convolution_and_modulus(_input, l, i_j1)
                for i_q, q in enumerate(integral_powers):
                    heatmap += (order_1_weights[i_q, i_j1, l] * conv_modulus**q)[...,0]
                if not order_2:
                    continue
                for i_j2 in range(i_j1+1, len(self.j_values)):
                    if order_2_integer_j and int(self.j_values[i_j2]) != self.j_values[i_j2]:
                        continue
                    conv_modulus_2 = convolution_and_modulus(conv_modulus, l, i_j2)
                    for i_q, q in enumerate(integral_powers):
                        heatmap += (order_2_weights[i_q, i_j2_weights, l] * conv_modulus_2**q)[...,0]
                    i_j2_weights += 1

        return heatmap


    def forward(self, input, fourier_input=False, order_2=True, operator='rotation_covariant_convolution', method='standard', method_args=None, order_2_integer_j=False):
        self._check_input(input, fourier_input=fourier_input)

        if operator == 'rotation_covariant_convolution':
            convolution_and_modulus = self._rotation_covariant_convolution_and_modulus
        elif operator == 'convolution':
            convolution_and_modulus = self._convolution_and_modulus
        else:
            raise ValueError('Unknow operator {}.'.format(operator))

        compute_scattering_coefs = self._compute_scattering_coefs


        if fourier_input:
            _input = input
            s_order_0 = compute_scattering_coefs(
                torch.abs(self.fft(_input, inverse=True)), method, method_args)
        else:
            _input = to_complex(input)
            s_order_0 = compute_scattering_coefs(_input, method, method_args)

        s_order_1 = []
        s_order_2 = []

        for l in range(self.L+1):
            s_order_1_l, s_order_2_l = [], []
            for i_j1 in range(len(self.j_values)):
                conv_modulus = convolution_and_modulus(_input, l, i_j1, fourier_input=fourier_input)
                s_order_1_l.append(compute_scattering_coefs(conv_modulus, method, method_args))
                if not order_2:
                    continue
                for i_j2 in range(i_j1+1, len(self.j_values)):
                    if order_2_integer_j and int(self.j_values[i_j2]) != self.j_values[i_j2]:
                        continue
                    conv_modulus_2 = convolution_and_modulus(conv_modulus, l, i_j2)
                    s_order_2_l.append(compute_scattering_coefs(conv_modulus_2, method, method_args))
            s_order_1.append(torch.cat(s_order_1_l, -1))
            if order_2:
                s_order_2.append(torch.cat(s_order_2_l, -1))

        if order_2:
            return s_order_0, torch.stack(s_order_1, dim=-1), torch.stack(s_order_2, dim=-1)
        return s_order_0, torch.stack(s_order_1, dim=-1)

    __call__ = forward
