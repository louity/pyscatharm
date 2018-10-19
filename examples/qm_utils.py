import numpy as np
import torch
from scipy.spatial.distance import pdist

try:
    from sklearn import linear_model, model_selection, preprocessing, pipeline, kernel_ridge
except ImportError as err:
    print('Error importing sklearn {}. '
        'You need to have sklearn installed to run qm7/9 example : '
        'pip install scikit-learn'.format(err))
    exit()
try:
    from cheml.datasets import load_qm7, load_qm9
except ImportError as err:
    print('Error importing cheml {}. '
        'You need to have cheml installed to run qm7/9 example : '
        'https://www.github.com/CheML/CheML'.format(err))
    exit()

def evaluate_regression(scat, target, cross_val_folds, kind='linear', alphas=10.**(-np.arange(0, 10))):
    for i, alpha in enumerate(alphas):
        if kind == 'linear':
            model = linear_model.Ridge(alpha=alpha)
        elif kind == 'bilinear':
            model = kernel_ridge.KernelRidge(alpha=alpha, kernel='poly', degree=2)
        else:
            raise ValueError('Invalid kind {}'.format(kind))
        regressor = pipeline.make_pipeline(preprocessing.StandardScaler(), model)
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


def get_qm_energies_and_folds(random_folds=False, database='qm7'):
    if database == 'qm7':
        qm = load_qm7(align=True)
        P = qm.P
    elif database == 'qm9':
        qm = load_qm9(align=True)
        P = qm.P_stratified_Ua.transpose()
    else:
        raise ValueError('only qm7 and qm9 databases')
    n_folds = P.shape[0]
    energies = qm.T.transpose()
    if random_folds:
        P = np.random.permutation(energies.shape[0]).reshape((n_folds, -1))
    cross_val_folds = []
    for i_fold in range(n_folds):
        fold = (np.concatenate(P[np.arange(n_folds) != i_fold], axis=0), P[i_fold])
        cross_val_folds.append(fold)
    return energies, cross_val_folds


def get_qm_positions_energies_and_charges(sigma, overlapping_precision, database='qm7'):
    if database == 'qm7':
        qm = load_qm7(align=True)
    elif database == 'qm9':
        qm = load_qm9(align=True)
    else:
        raise ValueError('only qm7 and qm9 databases')
    positions = qm.R
    atomic_numbers = qm.Z
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
