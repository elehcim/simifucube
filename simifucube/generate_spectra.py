# This file is part of Simifucube.
#
# Copyright (C) 2019 Michele Mastropietro (michele.mastropietro@gmail.com)
#
# Simifucube is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simifucube is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simifucube.  If not, see <https://www.gnu.org/licenses/>.

import gc
import pickle

import astropy.units as u
import numpy as np
import pynbody
import tqdm
from simulation.angmom import sideon  # TODO use pynbody, so that I remove the dependency on simulation
from specutils import Spectrum1D

from simifucube.spectra import Spectrum
from simifucube.util.congrid import congrid

MUSE_LIMITS = {'start': 4750, 'stop': 9600, 'step': 1.25}
MUSE_WVL = np.arange(**MUSE_LIMITS)

L_sol_in_erg_per_s = 3.839e33  # erg s-1
kpc_in_cm = 3.086e+21  # cm

conv_fact = L_sol_in_erg_per_s / (4 * np.pi * kpc_in_cm**2)  # Lsol cm-2 ~ erg s-1 cm-2
print('conv_fact=',conv_fact)
def compute_intensity(spectra, filt):
    intensity = list()
    for sp in spectra:
        intensity.append(sp.convolve(filt))  # erg s-1 cm-2
    return intensity


def kpc_to_arcsec(kpc, dist):
    return kpc / dist * (3600 * 180)/np.pi


def rebin(sp, new_bins):
    wvl = sp.wavelength.value
    limit = np.where(np.logical_and(wvl > new_bins.min(), wvl < new_bins.max()))[0]
    limited_fl = sp.flux[:, limit].value
    nstar = sp.shape[0]
    n_channels = len(new_bins)
    new_fl = congrid(limited_fl, (nstar, n_channels))
    new_sp = Spectrum1D(spectral_axis=new_bins * sp.spectral_axis_unit, flux=new_fl * sp.flux.unit)
    return new_sp


# def rebin_muse(sp):
#     new_bins_limits = {}
#     wvl = sp.wavelength.value
#     new_bins_limits['start'] = max(wvl.min(), MUSE_LIMITS['start'])
#     new_bins_limits['stop'] = min(wvl.max(), MUSE_LIMITS['stop'])
#     new_bins_limits['step'] = MUSE_LIMITS['step']
#     print(new_bins_limits)
#     new_bins = np.arange(**new_bins_limits)
#     rebinned = rebin(sp, new_bins=new_bins)
#     return rebinned

def cut_spectra(sp_list, cut_freq):
    """
    Cut the spectra removing the region greater than cut_freq

    Spectral axis of the sp_list should be identical for all the spectra
    """
    idx = np.digitize(cut_freq, sp_list[0].wavelength)
    print("Cutting spectra at {:.2f} ({})".format(cut_freq, idx))
    new_sp_list = list()
    for sp in sp_list:
        new_sp = Spectrum(sp.wavelength[:idx], sp.flux[:idx])
        new_sp_list.append(new_sp)
    return new_sp_list


STD_MET = -1.5
STD_AGE = 3

# STD_MULTIPLE_AGE = [1, 4]

def generate_spectra(snap, z_dist=20000, doppler_shift=True, use_template_star=False):
    """This is the core function allowing a set of particles in a pynbody snap to be associated with a list of spectra.
    Distance and redshift are taken into account.
    """
    print("Generating spectra...")
    print('Redshift: {:.2}'.format(snap.ancestor.header.redshift))
    print('Distance: {:.0f} Mpc'.format(z_dist/1000.0))
    if doppler_shift:
        print('Doppler-shift requested')
    if use_template_star:
        print('using fixed metallicity ({}) and age ({})'.format(STD_MET, STD_AGE))
    whole_spectrum_list = list()
    pos_list = list()

    last_valid_freq = np.inf
    # Here I cannot use directly the iteration over the array snap.s because
    # by default it iterates over the fields available ['pos', 'vx', ...]
    with snap.immediate_mode:
        for i in tqdm.tqdm(range(len(snap.s))):
            star = snap.s[i]
            pos = star['pos'][0].view(np.ndarray)
            pos_list.append(pos)
            dist_sq = np.linalg.norm(pos - np.array([0, 0, z_dist]))**2

            # print(star)
            if use_template_star:
                sp1 = Spectrum.from_met_age(STD_MET, STD_AGE)
                # sp1 = Spectrum.from_met_age(STD_MET, STD_MULTIPLE_AGE[i])
            else:
                sp1 = Spectrum.from_star(star)

            if doppler_shift:
                sp = sp1.doppler_shifted(star['vz'])
            else:
                sp = sp1

            if np.count_nonzero(sp.flux) != len(sp):
                last_nonzero_idx = np.nonzero(sp.flux)[0][-1]
                last_nonzero_freq = sp.wavelength[last_nonzero_idx]
                if last_nonzero_freq < last_valid_freq:
                    last_valid_freq = last_nonzero_freq

            mass = star['mass'].in_units('Msol')
            # print("mass (Msol)", mass)
            # sp.flux *= mass * L_sol_in_erg_per_s / (4 * np.pi * dist_sq * kpc_in_cm**2)  # erg s-1 cm-2 A-1
            sp.flux *= mass * conv_fact / dist_sq  # L_sol Msol-1 A-1 * Lsol/erg s-1 cm-2 A-1

            # print('flux', sp.flux)
            whole_spectrum_list.append(sp)

            # del sp
            # if i % 1000 == 0:
            #   gc.collect()

            # spectra.append(sp)

    spectra_list = cut_spectra(whole_spectrum_list, last_valid_freq)


    del sp1
    del whole_spectrum_list
    gc.collect()

    x = list()
    y = list()

    # positions in arcseconds
    for p in pos_list:
        x.append(kpc_to_arcsec(p[0], z_dist))
        y.append(kpc_to_arcsec(p[1], z_dist))
    positions = (np.array(x), np.array(y))

    # smoothing lengths
    if 'boxsize' in snap.properties:
        del snap.properties['boxsize']
    smooth = snap.s['smooth'].view(np.ndarray)
    return spectra_list, positions, smooth, last_valid_freq


def to_spectrum1d(spectra_list):
    flux = u.Quantity(np.vstack([sp.flux for sp in spectra_list]), unit='erg s-1 cm-2 AA-1')
    wavelength = u.Quantity(spectra_list[0].wavelength, unit='AA')
    my_spectrum = Spectrum1D(spectral_axis=wavelength, flux=flux.to('1e-20 erg s-1 cm-2 AA-1'))
    return my_spectrum


def snap_preprocess(snap, size_cuboid):
    """
    We center, put edge-on and select star particles from a cuboid of a certain edge size.
    The edgeon rotation is done on a sphere of radius sqrt(3) the cuboid size (so at least all the
    particles inside the cuboid are used to compute angmom)
    """
    # center on the stars
    print("Centering on stars")
    pynbody.analysis.halo.center(snap.s)
    print("R_eff = {:.2f} kpc".format(pynbody.analysis.luminosity.half_light_r(snap.s[pynbody.filt.Sphere(10)])))

    radius = size_cuboid * np.sqrt(3)
    print("Selecting a sphere of radius {:.2} kpc".format(radius))

    sphere = pynbody.filt.Sphere(radius)

    s_sphere = snap[sphere]

    # pynbody.analysis.angmom.sideon(s_sphere.s)
    print("Computing angular momentum...")
    sideon(s_sphere.s, cen=(0,0,0))
    print("Selecting a cuboid size {:} kpc".format(size_cuboid))

    cuboid = pynbody.filt.Cuboid(-size_cuboid)
    s = s_sphere[cuboid]
    print("R_eff = {:.2f} kpc".format(pynbody.analysis.luminosity.half_light_r(s)))
    return s


def spectra_from_snap(snap_name, size_cuboid, doppler_shift=True, use_fix_val=False):
    print('Loading {}'.format(snap_name))
    snap = pynbody.load(snap_name)
    s = snap_preprocess(snap, size_cuboid)
    print("Found {} stars".format(len(s.s)))
    sp_list, pos, last_valid_freq = generate_spectra(s, doppler_shift=doppler_shift, use_fix_val=use_fix_val)
    return to_spectrum1d(sp_list), pos, last_valid_freq


def spectra_from_pickle(pickle_name):
    sp_list, pos, last_valid_freq = pickle.load(open(pickle_name, "rb"))
    return to_spectrum1d(sp_list), pos, last_valid_freq



class SnapSpectra:
    def __init__(self, snap_name, size_cuboid, do_preprocessing=True):
        self.snap_name = snap_name
        self._size_cuboid = size_cuboid
        self.spectrum = None
        self.pos = None
        self.last_valid_freq = None
        # Do the actual initialization
        print('Loading {}'.format(snap_name))
        if pynbody.__version__ == '0.46':
            snap = pynbody.load(snap_name)#, ignore_cosmo=True)
        else:
            print("Ignoring cosmology (pynbody version {})".format(pynbody.__version__))
            snap = pynbody.load(snap_name, ignore_cosmo=True)
        snap.properties['boxsize'] = 1000 * pynbody.units.kpc
        print("boxsize set to:", snap.properties['boxsize'])
        if do_preprocessing:
            print('Preprocessing snap...')
            self._preprocessed_snap = snap_preprocess(snap, size_cuboid)
        else:
            print('Not preprocessing snap')
            self._preprocessed_snap = snap
        print("Found {} stars".format(len(self._preprocessed_snap.s)))

    def generate_spectra(self, save_as=None, doppler_shift=True, use_template_star=False):
        sp_list, pos, smooth, last_valid_freq = generate_spectra(self._preprocessed_snap, doppler_shift=doppler_shift, use_template_star=use_template_star)
        self.spectrum = to_spectrum1d(sp_list)
        del sp_list
        self.pos = pos
        self.last_valid_freq = last_valid_freq
        if save_as is not None:
            self.to_pickle(save_as)

    def generate_spectra_from_pickle(self, pickle_name):
        print('Getting spectra from', pickle_name)
        data = pickle.load(open(pickle_name, "rb"))
        n_spectra = len(data['flux'])
        print('Got {} spectra...'.format(n_spectra))
        assert n_spectra == self.n_stars
        self.spectrum = Spectrum1D(spectral_axis=data['wvl'], flux=data['flux'])
        self.pos = data['pos']
        self.last_valid_freq = data['last_valid_freq']

    def __repr__(self):
        return "Spectrum from file {} with {} stars".format(self.snap_name, self.n_stars)

    @property
    def n_stars(self):
        return len(self._preprocessed_snap.stars)

    def _metadata(self):
        pass

    def to_pickle(self, save_as):
        print('Saving spectra as', save_as)
        pickle.dump(dict(flux=self.spectrum.flux,
                         wvl=self.spectrum.wavelength,
                         pos=self.pos,
                         last_valid_freq=self.last_valid_freq), open(save_as, "wb"))


    def to_fits(self, save_as):
        raise NotImplementedError
        print('Saving spectra as', save_as)
        pickle.dump((self.spectra_list, self.positions, self.last_valid_freq), open(save_as, "wb"))


def muse_rebin(last_valid_freq, cube):
    print("Adapting spectra to MUSE range and spectral resolution...")
    if last_valid_freq > MUSE_LIMITS['stop']:
        last_valid_freq = MUSE_LIMITS['stop']
    new_bins = np.arange(MUSE_LIMITS['start'], last_valid_freq, MUSE_LIMITS['step']) * u.AA
    print('new_bins=',new_bins)
    muse_cube = cube.spectral_interpolate(new_bins)
    return muse_cube