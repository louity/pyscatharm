import numpy as np
import torch
from scatharm.utils import generate_weighted_sum_of_gaussians
from scatharm import SolidHarmonicScattering
from qm_utils import  get_qm_positions_energies_and_charges
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='QM database molecule density viewer')
parser.add_argument('--database', default='qm7', type=str, choices=['qm7', 'qm9'],
                    help='database to use (default: qm7)')
parser.add_argument('--molecule', '-p', default=0, type=int,
                    help='index of the molecule in the database (default: 0)')


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
    ax.set_title(ax.index)

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

args = parser.parse_args()
database = args.database
cuda = torch.cuda.is_available()
molecule_index = args.molecule

M, N, O = 192, 128, 96
grid = torch.from_numpy(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'))

overlapping_precision = 1e-1
sigma = 2.
pos, atomic_numbers, atom_valences, electron_valences = get_qm_positions_energies_and_charges(
        sigma, overlapping_precision, database=database)

if cuda:
    grid = grid.cuda()
    pos = pos.cuda()
    atomic_numbers = atomic_numbers.cuda()
    atom_valences = atom_valences.cuda()
    electron_valences = electron_valences.cuda()

start, end = molecule_index, molecule_index+1
molecule_positions = pos[start:end]

molecule_atomic_numbers = atomic_numbers[start:end]
molecule_atomic_number_density = generate_weighted_sum_of_gaussians(
        grid, molecule_positions, molecule_atomic_numbers, sigma, cuda=cuda)

molecule_atom_valences = atom_valences[start:end]
molecule_atom_valence_density = generate_weighted_sum_of_gaussians(
        grid, molecule_positions, molecule_atom_valences, sigma, cuda=cuda)

molecule_electron_valences = electron_valences[start:end]
molecule_electron_valence_density = generate_weighted_sum_of_gaussians(
        grid, molecule_positions, molecule_electron_valences, sigma, cuda=cuda)

molecule_core_density = molecule_atomic_number_density - molecule_electron_valence_density

# multi_slice_viewer(molecule_atomic_number_density[0].cpu().numpy())

j_values = [0, 0.5, 1, 1.5, 2]
L = 4
sigma = 2.
integral_powers = [1.0, 2.0, 3.0, 4.0]
basename = '4Ch_js_{}_L{}_Sig{}_IntPow_{}.npy'.format(
        j_values, L, sigma, integral_powers)

np_order_0 = np.load('./data/qm7/scat_0_' + basename).astype('float32')
np_order_1 = np.load('./data/qm7/scat_1_' + basename).astype('float32')
np_order_2 = np.load('./data/qm7/scat_2_' + basename).astype('float32')

order_0_shape = np_order_0.shape[1:]
order_1_shape = np_order_1.shape[1:]
order_2_shape = np_order_2.shape[1:]

n_order_0 = np.prod(order_0_shape)
n_order_1 = np.prod(order_1_shape)
n_order_2 = np.prod(order_2_shape)

scattering_coef = np.concatenate([
    np_order_0.reshape((7165, -1)),
    np_order_1.reshape((7165, -1)),
    np_order_2.reshape((7165, -1))], axis=1)
target = np.load('./data/qm7/energies.npy')


n_datapoints = scattering_coef.shape[0]
X = scattering_coef.copy()
y = target.copy()
scaler = preprocessing.StandardScaler(with_mean=False)
scaler.fit(X)
X_scaled = scaler.transform(X)

n_folds = 5
np.random.seed(0)
perm = np.random.permutation(n_datapoints)
train_indices = perm[:(4*n_datapoints)//5]
test_indices = perm[(4*n_datapoints)//5:]
alpha = 1e-6
regressor = linear_model.Ridge(alpha=alpha)
X_tr , y_tr = X_scaled[train_indices], y[train_indices]
X_te , y_te = X_scaled[test_indices], y[test_indices]

regressor.fit(X_tr, y_tr)
y_te_pred = regressor.predict(X_te)

MAE = np.mean(np.abs(y_te_pred - y_te))
RMSE = np.sqrt(np.mean((y_te_pred - y_te)**2))
print('Ridge regression, MAE: {}, RMSE: {}'.format(
    MAE, RMSE))

coef = (regressor.coef_ / scaler.scale_).astype('float32')
order_0_weights = torch.from_numpy(coef[:n_order_0].reshape(order_0_shape)).cuda()
order_1_weights = torch.from_numpy(coef[n_order_0:n_order_0+n_order_1].reshape(order_1_shape)).cuda()
order_2_weights = torch.from_numpy(coef[n_order_0+n_order_1:].reshape(order_2_shape)).cuda()

bias = regressor.intercept_
print(target[test_indices][0], X[test_indices][0].dot(coef) + bias)


M, N, O = 192, 128, 96
print('Creating scattering object...')
scat = SolidHarmonicScattering(M=M, N=N, O=O, j_values=j_values, L=L, sigma_0=sigma)
print('Scattering object created')

atomic_number_heatmap = scat.compute_heatmap(molecule_atomic_number_density, integral_powers, order_0_weights[...,0], order_1_weights[...,0],
            order_2_weights=order_2_weights[...,0], order_2=True, order_2_integer_j=False)

multi_slice_viewer(atomic_number_heatmap[0].cpu().numpy())
