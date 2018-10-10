import numpy as np
import torch
from tqdm import tqdm
from cheml.datasets import load_qm7
from sklearn import linear_model, model_selection, preprocessing, pipeline
from scipy.spatial.distance import pdist
from scatharm.scattering import SolidHarmonicScattering
from scatharm.utils import generate_weighted_sum_of_gaussians_in_fourier_space


def evaluate_linear_regression(scat, target, cross_val_folds, alphas=10.**(-np.arange(0, 10))):
    for i, alpha in enumerate(alphas):
        regressor = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.Ridge(alpha=alpha))
        scat_prediction = model_selection.cross_val_predict(regressor, X=scat, y=target, cv=cross_val_folds)
        scat_MAE = np.mean(np.abs(scat_prediction - target))
        scat_RMSE = np.sqrt(np.mean((scat_prediction - target)**2))
        print('Ridge regression, alpha: {}, MAE: {}, RMSE: {}'.format(
            alpha, scat_MAE, scat_RMSE))


def get_electron_valences(atomic_numbers):
    assert np.all(atomic_numbers >= 0), "atomic number must be positive"
    assert np.all(atomic_numbers < 19), "valence electrons computation not implemented for atomic number gt 18"
    return (
        atomic_numbers * (atomic_numbers <= 2) +
        (atomic_numbers - 2) * np.logical_and(atomic_numbers > 2, atomic_numbers <= 10) +
        (atomic_numbers - 10) * np.logical_and(atomic_numbers > 8, atomic_numbers <= 18))


def get_atom_valences(atomic_numbers):
    #FIXME: arbitrary choices have been made for N (valence max 5) and Cl (valence max 7)
    assert np.all(atomic_numbers >= 0), "atomic number must be positive"
    return (
        np.isin(atomic_numbers, [1, 3, 9, 17, 11, 19, 37, 55, 87]) * 1 + # H, Li, F, Cl, Na, K, Rb, Cs, Fr
        np.isin(atomic_numbers, [4, 8, 12, 20, 38]) * 2 + # Be, O, Mg, Ca, Sr
        np.isin(atomic_numbers, [5, 7, 13, 21, 31, 39]) * 3 + # B, N, Al, Sc, Ga, Y
        np.isin(atomic_numbers, [6, 14, 22, 32, 40, 50, 72, 82]) * 4 + # C, Si, Ti, Ge, Zr, Sn, Hf, Pb
        np.isin(atomic_numbers, [15, 23, 33]) * 5 + # P, V, As
        np.isin(atomic_numbers, [16, 24, 34]) * 6 + # S, Cr, Se
        np.isin(atomic_numbers, [35, 53]) * 7 # Br, I
    )


def get_sigma_0_wavelet(sigma, overlapping_precision):
    return sigma * np.sqrt(-2 * np.log(overlapping_precision) - 1)


def get_qm7_energies_and_folds(random_folds=False):
    qm7 = load_qm7(align=True)
    energies = qm7.T.transpose()
    n_folds = qm7.P.shape[0]
    if random_folds:
        P = np.random.permutation(energies.shape[0]).reshape((n_folds, -1))
    else:
        P = qm7.P
    cross_val_folds = []
    for i_fold in range(n_folds):
        fold = (np.concatenate(P[np.arange(n_folds) != i_fold], axis=0), P[i_fold])
        cross_val_folds.append(fold)
    return energies, cross_val_folds


def get_qm7_positions_energies_and_charges(M, N, O, J, L, sigma, overlapping_precision):
    qm7 = load_qm7(align=True)
    positions = qm7.R
    atomic_numbers = qm7.Z
    atom_valences = get_atom_valences(atomic_numbers)
    electron_valences = get_electron_valences(atomic_numbers)

    min_dist = np.inf
    for i in range(positions.shape[0]):
        n_atoms = np.sum(atomic_numbers[i] != 0)
        distances = pdist(positions[i, :n_atoms, :])
        min_dist = min(min_dist, distances.min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    print(delta, min_dist)

    positions *= delta / min_dist

    return (
            torch.from_numpy(positions), torch.from_numpy(atomic_numbers),
            torch.from_numpy(atom_valences), torch.from_numpy(electron_valences),
        )


# def main():
    # """Trains a simple linear regression model with solid harmonic
    # scattering coefficients on the atomisation energies of the QM7
    # database.

    # Achieves a MAE of ... kcal.mol-1
    # """

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
J, L = 2, 3
integral_powers = [0.5, 1., 2., 3.]
args = {'integral_powers': integral_powers}
sigma_0 = get_sigma_0_wavelet(sigma, overlapping_precision)
pos, atomic_numbers, atom_valences, electron_valences = get_qm7_positions_energies_and_charges(
        M, N, O, J, L, sigma, overlapping_precision)

n_molecules = pos.size(0)
n_batches = np.ceil(n_molecules / batch_size).astype(int)

scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma_0)

scat_0, scat_1, scat_2 = [], [], []
print('Computing solid harmonic scattering coefficients of molecules '
      'of QM7 database on {}'.format('GPU' if cuda else 'CPU'))
print('batch_size: {}, n_batches: {}'.format(batch_size, n_batches))
print('L: {}, J: {}, integral powers: {}'.format(L, J, integral_powers))

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

    # # atom valence batch
    # atom_valence_batch = atom_valences[start:end]
    # atom_valence_density_batch = generate_weighted_sum_of_gaussians(
            # grid, pos_batch, atom_valence_batch, sigma, cuda=cuda)
    # atom_valence_scat_0 = compute_integrals(atom_valence_density_batch, integral_powers)
    # atom_valence_scat_1, atom_valence_scat_2 = scat(atom_valence_density_batch, order_2=True,
                                  # method='integral', method_args=args)

    # # electron valence batch
    electron_valence_batch = electron_valences[start:end]
    electron_valence_fourier_density_batch = generate_weighted_sum_of_gaussians_in_fourier_space(
            fourier_grid, pos_batch, electron_valence_batch, sigma, cuda=cuda)
    electron_valence_scat_0, electron_valence_scat_1, electron_valence_scat_2 = scat(
            electron_valence_fourier_density_batch, fourier_input=True,
            order_2=True, method='integral', method_args=args)

    # # old core channel
    core_fourier_density_batch = atomic_number_fourier_density_batch - electron_valence_fourier_density_batch
    core_scat_0, core_scat_1, core_scat_2 = scat(
            core_fourier_density_batch, fourier_input=True, order_2=True,
            method='integral', method_args=args)

    scat_0.append(torch.cat([atomic_number_scat_0, electron_valence_scat_0, core_scat_0], dim=-1))

    scat_1.append(torch.stack([atomic_number_scat_1, electron_valence_scat_1, core_scat_1], dim=-1))
    scat_2.append(torch.stack([atomic_number_scat_2, electron_valence_scat_2, core_scat_2], dim=-1))

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

energies, cross_val_folds = get_qm7_energies_and_folds()
evaluate_linear_regression(scat_0_1_2, energies, cross_val_folds)

# if __name__ == '__main__':
    # main()
