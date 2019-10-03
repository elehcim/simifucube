import os
import numpy as np
from generate_spectra import spectra_from_snap, spectra_from_pickle
import pynbody
from create_noisy_cube import create_cube, contract_name
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt


if __name__ == '__main__':

    sim = 69002
    peri = 200
    isnap = 227
    # snap_name = "/media/michele/My Book/Michele/MySimulations/MovingBox/np/ongoing/mb.69002_p200_a800_r600/out/snapshot_{:04d}".format(isnap)
    # snap_name = "/media/michele/My Book/Michele/MySimulations/MovingBox/np/ongoing/mb.{}_p{}_a800_r600/out/snapshot_{:04d}".format(sim, peri, isnap)
    snap_name = "/home/michele/sim/MySimulations/ng/mb.{}_p{}_a800_r600/out/snapshot_{:04d}".format(sim, peri, isnap)
    print('Star mass: {:.2g} Msol'.format(pynbody.load(snap_name).s['mass'].sum().in_units('Msol')))
    force = False
    overwrite_output=True

    size_cuboid = 3

    save_pickle = contract_name("sp", sim, isnap, peri, size_cuboid, 'pkl')
    # save_pickle = "sp_{}p{:g}_{:04d}_r{}.pkl".format(sim_str, isnap, size_cuboid)
    # save_pickle = "sp_69p2_0300_everything.pkl"
    bins = 80
    masked = False


    if not force and os.path.isfile(save_pickle):
        print('Reading from file: {}'.format(save_pickle))
        spectra1d, (x,y), last_valid_freq = spectra_from_pickle(save_pickle)
    else:
        print('Getting spectra from ', snap_name)
        spectra1d, (x,y), last_valid_freq = spectra_from_snap(snap_name, save_as=save_pickle, size_cuboid=size_cuboid)

    a, cube = create_cube(spectra1d, x, y, bins, masked)

    # Diagnostics
    im, _, _, _ = plt.hist2d(x,y, bins)
    plt.colorbar()
    plt.title('Histogram2D of star particles per bin')
    fig = plt.figure()
    plt.hist(im.flatten())
    plt.yscale('log')
    print("tot: {:d}  mean: {:.2f}  median: {:.2f}".format(int(im.sum()), im.mean(), np.median(im)))
    plt.title('Histogram of star particles per bin')


    moment_0 = cube.moment(order=0)
    moment_1 = cube.linewidth_sigma()#moment(order=1)
    moment_2 = cube.moment(order=2)
    moment_0.quicklook('moment_0.png')
    moment_1.quicklook('moment_1.png')
    moment_2.quicklook('moment_2.png')

    if masked:
        cube[:, bins//2, bins//2].quicklook()
        plt.show()

    cube_name = contract_name("cube", sim, isnap, peri, size_cuboid, 'fits', bins=bins)
    print("writing", cube_name)
    cube.write(cube_name, format='fits', overwrite=overwrite_output)
    plt.show()