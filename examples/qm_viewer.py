import numpy as np
import torch
from scatharm.utils import generate_weighted_sum_of_gaussians
from qm_utils import  get_qm_positions_energies_and_charges
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

def main(args):
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
        # atom_valences = atom_valences.cuda()
        # electron_valences = electron_valences.cuda()


    molecule_positions = pos[molecule_index:molecule_index+1]
    molecule_atomic_numbers = atomic_numbers[molecule_index:molecule_index+1]

    molecule_atomic_number_density = generate_weighted_sum_of_gaussians(
            grid, molecule_positions, molecule_atomic_numbers, sigma, cuda=cuda)[0].cpu().numpy()

    multi_slice_viewer(molecule_atomic_number_density)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
