import numpy as np
import torch
from tqdm import tqdm
from cheml.datasets import load_qm7
from sklearn import linear_model, model_selection
from scipy.spatial.distance import pdist
from scatharm.scattering import SolidHarmonicScattering
from scatharm.utils import compute_integrals


def evaluate_bilinear_regression(scat_0, scat_1, scat_2, target):
    x = np.concatenate([scat_0, scat_1, scat_2], axis=1)
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


def evaluate_linear_regression(scat_0, scat_1, scat_2, target):
    x_1 = np.concatenate([scat_0, scat_1], axis=1)
    x_1_2 = np.concatenate([x_1, scat_2], axis=1)
    lin_regressor = linear_model.LinearRegression()
    scat_1_prediction = model_selection.cross_val_predict(lin_regressor, x_1, target)
    scat_1_MAE = np.mean(np.abs(scat_1_prediction - target))
    scat_1_RMSE = np.sqrt(np.mean((scat_1_prediction - target)**2))
    print('''scattering order 1, linear regression,
          MAE: {}, RMSE: {} (kcal.mol-1)'''.format(scat_1_MAE, scat_1_RMSE))

    scat_1_2_prediction = model_selection.cross_val_predict(lin_regressor, x_1_2, target)
    scat_1_2_MAE = np.mean(np.abs(scat_1_2_prediction - target))
    scat_1_2_RMSE = np.sqrt(np.mean((scat_1_2_prediction - target)**2))
    print('''scattering order 1 and 2, linear regression,
          MAE: {}, RMSE: {} (kcal.mol-1)'''.format(scat_1_2_MAE, scat_1_2_RMSE))


def get_valence(charges):
    return (
        charges * (charges <= 2) +
        (charges - 2) * np.logical_and(charges > 2, charges <= 10) +
        (charges - 10) * np.logical_and(charges > 8, charges <= 18))


def renormalize(positions, charges, sigma, overlapping_precision=1e-1):
    min_dist = np.inf
    for i in range(positions.shape[0]):
        n_atoms = np.sum(charges[i] != 0)
        pos = positions[i, :n_atoms, :]
        min_dist = min(min_dist, pdist(pos).min())

    delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))

    return positions * delta / min_dist


def get_qm7_positions_energies_and_charges(M, N, O, J, L, sigma):
    qm7 = load_qm7(align=True)
    positions = qm7.R
    charges = qm7.Z
    energies = qm7.T.transpose()
    valence_charges = get_valence(charges)

    positions = renormalize(positions, charges, sigma)

    return torch.from_numpy(positions), torch.from_numpy(energies), torch.from_numpy(charges), torch.from_numpy(valence_charges)


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


def main():
    """Trains a simple linear regression model with solid harmonic
    scattering coefficients on the atomisation energies of the QM7
    database.

    Achieves a MAE of ... kcal.mol-1
    """
    cuda = torch.cuda.is_available()
    batch_size = 32
    M, N, O = 192, 128, 96
    grid = torch.from_numpy(
            np.fft.ifftshift(np.mgrid[-M//2:-M//2+M, -N//2:-N//2+N, -O//2:-O//2+O].astype('float32'), axes=(1,2,3)))
    if cuda:
        grid = grid.cuda()
    sigma = 2.
    J, L = 2, 3
    integral_powers = [0.5, 1., 2., 3.]
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
        full_scat_1, full_scat_2 = scat(full_density_batch, True, 'integral', args)

        val_batch = valence_charges[start:end]
        val_density_batch = generate_weighted_sum_of_gaussians(
                grid, pos_batch, val_batch, sigma, cuda=cuda)
        val_scat_0 = compute_integrals(val_density_batch, integral_powers)
        val_scat_1, val_scat_2 = scat(val_density_batch, True, 'integral', args)

        core_density_batch = full_density_batch - val_density_batch
        core_scat_0 = compute_integrals(core_density_batch, integral_powers)
        core_scat_1, core_scat_2 = scat(core_density_batch, True, 'integral', args)

        scat_0.append(torch.stack([full_scat_0, val_scat_0, core_scat_0], dim=-1))
        scat_1.append(torch.stack([full_scat_1, val_scat_1, core_scat_1], dim=-1))
        scat_2.append(torch.stack([full_scat_2, val_scat_2, core_scat_2], dim=-1))

    n = batch_size * n_batches

    scat_0 = torch.cat(scat_0, dim=0)
    scat_1 = torch.cat(scat_1, dim=0)
    scat_2 = torch.cat(scat_2, dim=0)

    np_scat_0 = scat_0.numpy().reshape((n, -1))
    np_scat_1 = scat_1.numpy().reshape((n, -1))
    np_scat_2 = scat_2.numpy().reshape((n, -1))

    print('order 1 : {} coef, order 2 : {} coefs'.format(np_scat_1.shape[1], np_scat_2.shape[1]))
    target = energies.numpy()[:n]
    evaluate_linear_regression(np_scat_0, np_scat_1, np_scat_2, target)
    evaluate_bilinear_regression(np_scat_0, np_scat_1, np_scat_2, target)

    basename = 'qm7_L_{}_J_{}_sigma_{}_MNO_{}_powers_{}.npy'.format(
            L, J, sigma, (M, N, O), integral_powers)
    np.save('scat_0_' + basename, np_scat_0)
    np.save('scat_1_' + basename, np_scat_1)
    np.save('scat_2_' + basename, np_scat_2)


if __name__ == '__main__':
    main()
