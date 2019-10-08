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

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import pynbody

from simifucube.util.doppler_shift import dopplerShift

import logging


def setup_logger(logger_level=logging.INFO, logger_name=None):

    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    # log_file_handler = logging.FileHandler(os.path.join(config_logger['log_dir'], config_logger['log_file']))
    # log_file_handler.setFormatter(formatter)

    stream_formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
    log_stream_handler = logging.StreamHandler()
    log_stream_handler.setFormatter(stream_formatter)

    logger = logging.getLogger(logger_name)
    # logger.addHandler(log_file_handler)
    logger.addHandler(log_stream_handler)
    logger.setLevel(logger_level)

    return logger

logger = setup_logger(logging.INFO)

# Metallicity [M/H] (Z)
bins_metallicity = [-2.27, -1.79, -1.49, -1.26, -0.96, -0.66, -0.35, -0.25, +0.06, +0.15, +0.26, +0.40]

# Age (Gyr)
bins_age = [00.03, 00.04, 00.05, 00.06, 00.07, 00.08, 00.09, 00.10, 00.15, 00.20,
00.25, 00.30, 00.35, 00.40, 00.45, 00.50, 00.60, 00.70, 00.80, 00.90, 01.00,
01.25, 01.50, 01.75, 02.00, 02.25, 02.50, 02.75, 03.00, 03.25, 03.50, 03.75, 04.00,
04.50, 05.00, 05.50, 06.00, 06.50, 07.00, 07.50, 08.00, 08.50, 09.00, 09.50, 10.00,
10.50, 11.00, 11.50, 12.00, 12.50, 13.00, 13.50, 14.00]


SPECTRA_DIR = '/home/michele/sim/IFU_images/library/EMILES_BASTI_BASE_CH_FITS'
ELEMENT_LIMIT = 10000  # For MUSE it's fine (10000 spectral pixel = 10678.9 Angstrom)


# -- From http://www.iac.es/proyecto/miles/pages/ssp-models.php
# The stellar population synthesis model predictions and are based on the code presented in Vazdekis et al. (2010).
# The spectra are stored in air wavelengths.
# The total mass of the SSP is 1Mo. The SSP spectra are given in units of L_λ / Lo Mo^{-1} Å^{-1}.

# -- From http://www.iac.es/proyecto/miles/pages/ssp-models/name-convention.php
# Please note that the relation between the total metallicity and iron metallicity is
#      [Fe/H]=[M/H]-0.75[Mg/Fe]
# Thus when selecting a scaled-solar model with total metallicity [M/H] then [Fe/H]=[M/H],
# whereas for the alpha-enhanced models with the same metallicity [Fe/H]=[M/H]-0.3.
# Finally for the "base" models it is assumed that [Fe/H]=[M/H], although this is only true at high metallicities.
# For low metallicities the input stars are alpha-enhanced and therefore when selecting a model with
# total metallicity [M/H] its [Fe/H] is lower.
# To summarize the base models follow the abundance pattern of The Galaxy.
# Note that these models are inconsistent at low metallicities because the adopted isochrones are always scaled-solar.

# As an example, if only models with total metallicity [M/H]=+0.06, 10 Gyr and bimodal IMF shape with slope 1.30 are requested, then the following set of models computed with BaSTI isochrones can  be downloaded:
# * Mbi1.30Zp0.06T10.0000_iTp0.00_Ep0.00   scaled-solar abundance ratio ([M/H]=[Fe/H]=+0.06)
# * Mbi1.30Zp0.06T10.0000_iTp0.40_Ep0.40   alpha-enhanced abundance ratio ([M/H]=+0.06; [Fe/H]=-0.24)
# * Mbi1.30Zp0.06T10.0000_iTp0.00_baseFe   base models where no consideration about the abundance ratio is being made.
#                                          The stars feeding the models are picked on the basis of their [Fe/H] metallicity,
#                                          and therefore these models follow the abundance pattern of The Galaxy.
# as well as base models computed with the Padova00 isochrones with nearly similar metallicity:
# * Mbi1.30Zp0.00T10.0000_iPp0.00_baseFe


def get_nearest(x, bins):
    return bins[np.abs(np.array(bins) - x).argmin()]


def get_bin(x, bins):
    n = np.digitize(x, bins)
    return bins[n-1], bins[n]


def get_filename(metallicity, age):
    """
    Get filename of the correct SSP by parsing the E-MILES library.

    I use E-MILES with Chabrier IMF with 1.3 slope and BaSTI isochrones.
    The metallicity is [Fe/H] of the star particle
    Age is in Gyr
    """
    met = get_nearest(metallicity, bins_metallicity)
    _age = get_nearest(age, bins_age)
    met_sign = 'p' if met >= 0 else 'm'
    name = 'Ech1.30Z{}{:3.2f}T{:07.4f}_iTp0.00_baseFe.fits'.format(met_sign, abs(met), _age)
    filename = os.path.join(SPECTRA_DIR, name)
    return filename


def get_spectrum(filename, n_limit=ELEMENT_LIMIT):
    """
    Get spectrum from the given filename. Limit the read to the first `n_limit` elements


    Parameters
    ----------
    filename: str
        The filename
    n_limit: int (default 10000)
        Number of elements to read from the spectrum from the beginning. This reduces the memory footprint.
        This useful because last spectral channels are usually outside of range for practical purpose.
        For MUSE the default of 10000 is fine. 10000 spectral pixels corresponds to an upper limit of 10678.9 Angstrom, MUSE arrives at 9600 Angstroms

    Returns
    -------
    wavelength:np.ndarray
        Wavelength in Angstrom
    flux: np.ndarray
        Spectral flux per unit mass (Luminosity per unit wavelength per unit mass) in Lo Mo^{-1} Å^{-1}
    """
    with fits.open(filename) as hdul:
        hdu = hdul[0]

        low_limit = hdu.header['CRVAL1']
        N = hdu.header['NAXIS1']
        delt = hdu.header['CDELT1']
        # logger.debug("CRVAL1 = {}".format(low_limit))
        # logger.debug("NAXIS1 = {}".format(N))
        # logger.debug("CDELT1 = {}".format(delt))
        high_limit = low_limit + N * delt
        flux = hdu.data[:n_limit].copy()  #* u.angstrom # FIXME

        del hdu.data  # allow the file to close

    wavelength = np.linspace(low_limit, high_limit, N)[:n_limit]  # FIXME * u.angstrom

    return wavelength, flux

# Obsolete
# def get_ML(age, metallicity, band='V'):
#     df = mass_to_light()
#     df['MH_rounded'] = df['[M/H]'].round(2)
#     row = df.query('Age == {} & "MH_rounded" == {}'.format(age, metallicity))
#     ml = row['(M/L){}'.format(band)].values
#     return ml

# # FIXME be sure to having taken the correct one. This seems not the correct one
# def mass_to_light():
#     return pd.read_csv('out_phot_CH_BASTI.txt', delim_whitespace=True)


MgFe_corr = -0.261299
FeH_corr = -2.756433

@pynbody.derived_array
def mgfe(snap):
    arr = np.log10(snap.s['mgst']/snap.s['fest']) - MgFe_corr
    arr[np.logical_or(snap.s['mgst'] == 0.0, snap.s['fest'] == 0.0)] = 0.471782
    return arr

@pynbody.derived_array
def feh(snap):
    arr = np.log10(snap.s['fest']/snap.s['mass']) - FeH_corr
    arr[np.logical_or(snap.s['fest'] == 0.0, snap.s['mass'] == 0.0)] = -98.0
    return arr

class Spectrum:
    filename = None
    v_doppler_shift = 0

    def __init__(self, wavelength, flux):
        """wavelength in Angstrom
        Flux in L_sol Msol-1 A-1
        """
        self.wavelength = wavelength
        self.flux = flux

    @classmethod
    def from_file(cls, filename):
        cls.filename = filename
        logger.debug(os.path.basename(filename))
        wvl, f = get_spectrum(filename)
        my_sp = cls(wvl, f)
        my_sp.filename = filename
        return my_sp

    @classmethod
    def from_met_age(cls, metallicity, age):
        filename = get_filename(metallicity, age)
        return cls.from_file(filename)

    @classmethod
    def from_star(cls, star):
        # FIXME .in_units('Gyr') causes a problem in pynbody
        gadget_time_in_gyr = 0.9778139512067809
        metallicity = star['feh']
        age = star['age'] * gadget_time_in_gyr
        # print("ID: {} [Fe/H] = {}, age = {}".format(star['iord'], metallicity, age))
        return cls.from_met_age(metallicity, age)

    def __len__(self):
        return len(self.wavelength)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.wavelength, self.flux, **kwargs)
        ax.set_xlabel('$\lambda$ [$\AA$]')
        ax.set_ylabel('$L_λ$')

    def doppler_shifted(self, v, **kwargs):
        # Return a new doppler shifted spectrum
        # v in km/s
        self.v_doppler_shift = v
        wvl, flux = dopplerShift(self.wavelength, self.flux, v, **kwargs)
        my_sp = Spectrum(wvl, flux)
        my_sp.filename = self.filename
        return my_sp

    def convolve(self, filter):
        return filter.convolve_spectrum(self)


Spectrum.doppler_shifted.__doc__ = dopplerShift.__doc__


if __name__ == '__main__':
    import sys
    snap_name = '/home/michele/sim/MySimulations/np_glass/mb.62002_p200_a800_r600/out/snapshot_0048'
    s = pynbody.load(snap_name)
    if len(sys.argv) > 1:
        istar = int(sys.argv[1])
    else:
        istar=23

    my_star = s.s[istar]

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.hist(s.s['feh'], bins=100)
    ax1.set_title('[Fe/H]')
    ax1.axvline(-5, color='r')
    ax2.hist(s.s['age'], bins=100)
    ax2.set_title('Age [{}]'.format(s.s['age'].units))
    # filename = get_filename(my_star['metals'], my_star['age'])
    # flux, wavelength = get_spectrum(filename)
    print("my_star ({}):".format(istar))
    print("Z = {}, age = {}".format(my_star['metals'], my_star['age']))
    sp = Spectrum.from_star(my_star)
    print('len(sp)=', len(sp))
    fig, ax = plt.subplots()
    sp.plot(ax=ax)

    guilty = s.s[s.s['iord'] == 1315277]

    # for i in tqdm.tqdm(range(len(s.s[600:800]))):
    #     fig, ax = plt.subplots()
    #     # if i % 100 == 0:
    #     #   print(i)
    #     star = s.s[i]
    #     sp = Spectrum.from_star(star)
    #     sp.plot(ax=ax)
    #     ax.set_title("({}) ID: {} Z = {}, age = {}".format(i, star['iord'], star['metals'], star['age']))

    # fig.legend()
    # sp1 = sp.doppler_shifted(10000)
    # sp1.plot(ax=ax)
    plt.show()
