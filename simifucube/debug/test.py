import pynbody

import numpy as np
import pickle
from spectra import Spectrum
from congrid import congrid
from specutils import Spectrum1D
from astropy import units as u
from generate_spectra import spectra_from_snap


def congrid_debug(snap_name):
    sp, pos, my_last_valid_freq = spectra_from_snap(snap_name)
    print('Last valid freq:', my_last_valid_freq)
    my_idx = 26
    import matplotlib.pyplot as plt
    plt.step(sp.spectral_axis, sp[my_idx].flux)

    new_bins_limits = {}
    wvl = sp.wavelength.value
    # if wvl.min() < MUSE_LIMITS['start']:
    new_bins_limits['start'] = max(wvl.min(), MUSE_LIMITS['start'])
    new_bins_limits['stop'] = min(wvl.max(), MUSE_LIMITS['stop'])
    new_bins_limits['step'] = MUSE_LIMITS['step']
    # if wvl.max() > MUSE_LIMITS['stop']:
    #     new_bins_limits['stop'] = wvl.max()
    print(new_bins_limits)
    new_bins = np.arange(**new_bins_limits)

    # rebinned = rebin(sp, new_bins=new_bins)
    # wvl = sp.spectral_axis.value
    limit = np.where(np.logical_and(wvl > new_bins.min(), wvl < new_bins.max()))[0]
    limited_fl = sp.flux[:, limit].value
    nstar = sp.shape[0]
    n_channels = len(new_bins)
    new_fl = congrid(limited_fl, (nstar, n_channels))
    new_sp = Spectrum1D(spectral_axis=new_bins * sp.spectral_axis_unit, flux=new_fl * sp.flux.unit)
    # return new_sp
    plt.step(new_sp.spectral_axis, new_sp[my_idx].flux)

    plt.show()

def mg_lines(snap_name):
    sp, pos, my_last_valid_freq = spectra_from_snap(snap_name)
    print('Last valid freq:', my_last_valid_freq)
    my_idx = 26
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()

    excerpt = (1, -1)
    for idx in range(sp.shape[0]):
        ax1.step(sp.spectral_axis, sp[idx].flux)
    ax1.set_xlim(5100, 5300)
    fig, ax2 = plt.subplots()
    # ax2.step(sp.spectral_axis, sp[26:31+1].flux.sum(axis=0))
    ax2.step(sp.spectral_axis, sp[slice(*excerpt)].flux.sum(axis=0))
    ax2.set_xlim(5100, 5300)
    plt.show()


if __name__ == '__main__':
    snap_name = '/home/michele/sim/MySimulations/np_glass/mb.62002_p200_a800_r600/out/snapshot_0048'

    import matplotlib.pyplot as plt
    # sp, pos, my_last_valid_freq = spectra_from_snap(snap_name)
    snap = pynbody.load(snap_name)
    # print('Last valid freq:', my_last_valid_freq)
    my_idx = 26
    fig, ax1 = plt.subplots()

    sp = Spectrum.from_star(snap.s[my_idx])
    sp.plot(ax=ax1)
    vel = snap.s[my_idx]['vz']
    print(vel)
    ds_sp = sp.doppler_shifted(vel)
    ds_sp.plot(ax=ax1)
    # excerpt = (1, -1)
    # for idx in range(sp.shape[0]):
    #     ax1.step(sp.spectral_axis, sp[idx].flux)
    ax1.set_xlim(5100, 5300)
    # fig, ax2 = plt.subplots()
    # # ax2.step(sp.spectral_axis, sp[26:31+1].flux.sum(axis=0))
    # ax2.step(sp.spectral_axis, sp[slice(*excerpt)].flux.sum(axis=0))
    # ax2.set_xlim(5100, 5300)

    # new_bins_limits = {}
    # wvl = sp.wavelength.value
    # # if wvl.min() < MUSE_LIMITS['start']:
    # new_bins_limits['start'] = max(wvl.min(), MUSE_LIMITS['start'])
    # new_bins_limits['stop'] = min(wvl.max(), MUSE_LIMITS['stop'])
    # new_bins_limits['step'] = MUSE_LIMITS['step']
    # # if wvl.max() > MUSE_LIMITS['stop']:
    # #     new_bins_limits['stop'] = wvl.max()
    # print(new_bins_limits)
    # new_bins = np.arange(**new_bins_limits)

    # # rebinned = rebin(sp, new_bins=new_bins)
    # # wvl = sp.spectral_axis.value
    # limit = np.where(np.logical_and(wvl > new_bins.min(), wvl < new_bins.max()))[0]
    # limited_fl = sp.flux[:, limit].value
    # nstar = sp.shape[0]
    # n_channels = len(new_bins)
    # new_fl = congrid(limited_fl, (nstar, n_channels))
    # new_sp = Spectrum1D(spectral_axis=new_bins * sp.spectral_axis_unit, flux=new_fl * sp.flux.unit)
    # # return new_sp
    # plt.step(new_sp.spectral_axis, new_sp[my_idx].flux)

    plt.show()
