import numpy as np
import torch
from tqdm import tqdm
from cheml.datasets import load_qm7
from sklearn import linear_model, model_selection
from scipy.spatial.distance import pdist
from scatharm.scattering import SolidHarmonicScattering
from scatharm.utils import compute_integrals, generate_weighted_sum_of_gaussians


def evaluate_bilinear_regression(scat_1, scat_2, target):
    x = np.concatenate([scat_1, scat_2], axis=1)
    x = (x - x.mean()) / x.std()
    y_factor = (target.max() - target.min())
    y = (target - target.min()) / y_factor
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

    x_train = torch.autograd.Variable(torch.from_numpy(x_train), requires_grad=False)
    x_test = torch.autograd.Variable(torch.from_numpy(x_test), requires_grad=False)
    y_train = torch.autograd.Variable(torch.from_numpy(y_train), requires_grad=False)
    y_test = torch.autograd.Variable(torch.from_numpy(y_test), requires_grad=False)

    n_x = x_train.size(1)
    bilinear = torch.nn.Bilinear(n_x, n_x, 1, bias=False)
    linear = torch.nn.Linear(n_x, 1)
    loss_fn = torch.nn.MSELoss(size_average=True)
    params = list(bilinear.parameters()) + list(linear.parameters())

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    for t in range(20000):
        y_train_pred = bilinear(x_train, x_train) + linear(x_train)
        loss = loss_fn(y_train_pred, y_train)
        if (t % 500) == 0:
            print(t, np.sqrt(loss.data[0])*y_factor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('train RMSE : ', np.sqrt(loss.data[0])*y_factor)

    y_test_pred = bilinear(x_test, x_test) + linear(x_test)
    loss = loss_fn(y_test_pred, y_test)
    print('test RMSE: ', np.sqrt(loss.data[0])*y_factor)


def evaluate_linear_regression(scat, target):
    alphas = 10.**(-np.arange(5, 15))
    maes = np.zeros_like(alphas)
    x_tr, x_te, y_tr, y_te = model_selection.train_test_split()
    for i, alpha in enumerate(alphas):
        ridge = linear_model.Ridge(alpha=alpha)
        ridge.fit(x_tr, y_tr)
        maes[i] = np.mean(np.abs(y_te - ridge.predict(x_te)))
    print('Ridge regression :')
    for i in range(len(alphas)):
        print(alphas[i], maes[i])


def get_valence(charges):
    return (
        charges * (charges <= 2) +
        (charges - 2) * np.logical_and(charges > 2, charges <= 10) +
        (charges - 10) * np.logical_and(charges > 8, charges <= 18))


def renormalize(positions, charges, sigma, overlapping_precision=1e-1):
    min_dist = np.inf

    for i in range(positions.shape[0]):
        # positions[i,:,0] -= 0.5 * (positions[i,:,0].min() + positions[i,:,0].max())
        # positions[i,:,1] -= 0.5 * (positions[i,:,1].min() + positions[i,:,1].max())
        # positions[i,:,2] -= 0.5 * (positions[i,:,2].min() + positions[i,:,2].max())
        n_atoms = np.sum(charges[i] != 0)
        pos = positions[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos).min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
    print(delta / min_dist)

    return positions * delta / min_dist


def get_qm7_positions_energies_and_charges(M, N, O, J, L, sigma):
    qm7 = load_qm7(align=True)
    positions = qm7.R
    charges = qm7.Z
    energies = qm7.T.transpose()
    valence_charges = get_valence(charges)

    positions = renormalize(positions, charges, sigma)

    return torch.from_numpy(positions), torch.from_numpy(energies), torch.from_numpy(charges), torch.from_numpy(valence_charges)


# def main():
    # """Trains a simple linear regression model with solid harmonic
    # scattering coefficients on the atomisation energies of the QM7
    # database.

    # Achieves a MAE of ... kcal.mol-1
    # """
cuda = torch.cuda.is_available()
batch_size = 32
M, N, O = 192, 128, 96
grid = torch.from_numpy(
        np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
if cuda:
    grid = grid.cuda()
sigma = 1.5
J, L = 2, 1
integral_powers = [1.]
args = {'integral_powers': integral_powers}
pos, energies, full_charges, valence_charges = get_qm7_positions_energies_and_charges(
        M, N, O, J, L, sigma)
n_molecules = pos.size(0)
n_batches = np.ceil(n_molecules / batch_size).astype(int)

scat = SolidHarmonicScattering(M=M, N=N, O=O, J=J, L=L, sigma_0=sigma)

scat_0, scat_1, scat_2 = [], [], []
print('Computing solid harmonic scattering coefficients of molecules \
        of QM7 database on {}'.format('GPU' if cuda else 'CPU'))
print('L: {}, J: {}, integral powers: {}'.format(L, J, integral_powers))
for i in tqdm(range(n_batches)):
    start, end = i*batch_size, min((i+1)*batch_size, n_molecules)

    pos_batch = pos[start:end]
    full_batch = full_charges[start:end]

    full_density_batch = generate_weighted_sum_of_gaussians(
            grid, pos_batch, full_batch, sigma, cuda=cuda)

    full_scat_0 = compute_integrals(full_density_batch, integral_powers)
    full_scat_1, full_scat_2 = scat(full_density_batch, order_2=True,
                                    method='integral', method_args=args)

    full_scat_0 = compute_integrals(full_density_batch, integral_powers)
    val_batch = valence_charges[start:end]
    val_density_batch = generate_weighted_sum_of_gaussians(
            grid, pos_batch, val_batch, sigma, cuda=cuda)
    val_scat_0 = compute_integrals(val_density_batch, integral_powers)
    val_scat_1, val_scat_2 = scat(val_density_batch, order_2=True,
                                  method='integral', method_args=args)

    core_density_batch = full_density_batch - val_density_batch
    core_scat_0 = compute_integrals(core_density_batch, integral_powers)
    core_scat_1, core_scat_2 = scat(core_density_batch, order_2=True,
                                    method='integral', method_args=args)

    scat_0.append(torch.stack([full_scat_0, val_scat_0, core_scat_0], dim=-1))
    scat_1.append(torch.stack([full_scat_1, val_scat_1, core_scat_1], dim=-1))
    scat_2.append(torch.stack([full_scat_2, val_scat_2, core_scat_2], dim=-1))

scat_0 = torch.cat(scat_0, dim=0)
scat_1 = torch.cat(scat_1, dim=0)
scat_2 = torch.cat(scat_2, dim=0)

np_scat_0 = scat_0.numpy().reshape((n_molecules, -1))
np_scat_1 = scat_1.numpy().reshape((n_molecules, -1))
np_scat_2 = scat_2.numpy().reshape((n_molecules, -1))

basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
        L, J, sigma, (M, N, O), integral_powers)
np.save('scat_0_' + basename, np_scat_0)
np.save('scat_1_' + basename, np_scat_1)
np.save('scat_2_' + basename, np_scat_2)

scat = np.concatenate([np_scat_0, np_scat_1, np_scat_2], axis=1)
target = energies.numpy()

print('order 1 : {} coef, order 2 : {} coefs'.format(np_scat_1.shape[1], np_scat_2.shape[1]))

evaluate_linear_regression(scat, target)

evaluate_bilinear_regression(np_scat_1, np_scat_2, target)


# if __name__ == '__main__':
    # main()
