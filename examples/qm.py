import numpy as np
import torch
from tqdm import tqdm
from scatharm.scattering import SolidHarmonicScattering
from scatharm.utils import generate_weighted_sum_of_gaussians_in_fourier_space

from qm_utils import  evaluate_regression, get_qm_energies_and_folds, get_qm_positions_energies_and_charges

# def main():
    # """Trains a simple linear regression model with solid harmonic
    # scattering coefficients on the atomisation energies of the QM7
    # database.

    # Achieves a MAE of ... kcal.mol-1
    # """

database = 'qm7'
cuda = torch.cuda.is_available()
batch_size = 16
M, N, O = 192, 128, 96

fourier_grid_np = np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32')
fourier_grid_np[0] *= 2*np.pi / M
fourier_grid_np[1] *= 2*np.pi / N
fourier_grid_np[2] *= 2*np.pi / O

fourier_grid = torch.from_numpy(np.fft.ifftshift(fourier_grid_np, axes=(1,2,3)))
if cuda:
    fourier_grid = fourier_grid.cuda()
overlapping_precision = 1e-1
sigma = 1.5
j_values, L = [0, 1, 2], 3
integral_powers = [0.5, 1., 2., 3.]
args = {'integral_powers': integral_powers}
pos, atomic_numbers, atom_valences, electron_valences = get_qm_positions_energies_and_charges(
        sigma, overlapping_precision, database=database)

n_molecules = pos.size(0)
n_batches = np.ceil(n_molecules / batch_size).astype(int)

scat = SolidHarmonicScattering(M=M, N=N, O=O, j_values=j_values, L=L, sigma_0=sigma)

scat_0, scat_1, scat_2 = [], [], []
print('Computing solid harmonic scattering coefficients of {} molecules '
      'of {} database on {}'.format(n_molecules, database, 'GPU' if cuda else 'CPU'))
print('batch_size: {}, n_batches: {}'.format(batch_size, n_batches))
print('L: {}, j_values: {}, integral powers: {}'.format(L, j_values, integral_powers))

for i in tqdm(range(n_batches)):
    start, end = i*batch_size, min((i+1)*batch_size, n_molecules)

    pos_batch = pos[start:end]

    # atomic number channel
    atomic_number_batch = atomic_numbers[start:end]
    atomic_number_fourier_density_batch = generate_weighted_sum_of_gaussians_in_fourier_space(
            fourier_grid, pos_batch, atomic_number_batch, sigma, cuda=cuda)
    atomic_number_scat_0, atomic_number_scat_1, atomic_number_scat_2 = scat(
            atomic_number_fourier_density_batch, fourier_input=True,
            order_2=True, method='integral', method_args=args)

    # atomic valence channel
    atom_valence_batch = atom_valences[start:end]
    atom_valence_density_batch = generate_weighted_sum_of_gaussians_in_fourier_space(
            fourier_grid, pos_batch, atom_valence_batch, sigma, cuda=cuda)
    atom_valence_scat_0, atom_valence_scat_1, atom_valence_scat_2 = scat(
            atom_valence_density_batch, fourier_input=True, order_2=True,
            method='integral', method_args=args)

    # electron valence channel
    electron_valence_batch = electron_valences[start:end]
    electron_valence_fourier_density_batch = generate_weighted_sum_of_gaussians_in_fourier_space(
            fourier_grid, pos_batch, electron_valence_batch, sigma, cuda=cuda)
    electron_valence_scat_0, electron_valence_scat_1, electron_valence_scat_2 = scat(
            electron_valence_fourier_density_batch, fourier_input=True,
            order_2=True, method='integral', method_args=args)

    # core electrons channel
    core_fourier_density_batch = atomic_number_fourier_density_batch - electron_valence_fourier_density_batch
    core_scat_0, core_scat_1, core_scat_2 = scat(
            core_fourier_density_batch, fourier_input=True, order_2=True,
            method='integral', method_args=args)

    scat_0.append(
        torch.cat([atomic_number_scat_0, atom_valence_scat_0, electron_valence_scat_0, core_scat_0], dim=-1))
    scat_1.append(
        torch.stack([atomic_number_scat_1, atom_valence_scat_1, electron_valence_scat_1, core_scat_1], dim=-1))
    scat_2.append(
        torch.stack([atomic_number_scat_2, atom_valence_scat_2, electron_valence_scat_2, core_scat_2], dim=-1))

scat_0 = torch.cat(scat_0, dim=0)
scat_1 = torch.cat(scat_1, dim=0)
scat_2 = torch.cat(scat_2, dim=0)

print('scattering coefficients :')
print('order 0, shape {} = (n_molecules, n_powers, n_channels)'.format(scat_0.size()))
print('order 1, shape {} = (n_molecules, n_powers, J+1, L+1, n_channels)'.format(scat_1.size()))
print('order 2, shape {} = (n_molecules, n_powers, J(J+1)/2, L+1, n_channels)'.format(scat_1.size()))

np_scat_0 = scat_0.numpy().reshape((end, -1))
np_scat_1 = scat_1.numpy().reshape((end, -1))
np_scat_2 = scat_2.numpy().reshape((end, -1))

scat_0_1_2 = np.concatenate([np_scat_0, np_scat_1, np_scat_2], axis=1)

energies, cross_val_folds = get_qm_energies_and_folds(database=database)
evaluate_regression(scat_0_1_2, energies, cross_val_folds)

# if __name__ == '__main__':
    # main()
