"""Adapted from SKIRT9"""
import numpy as np
from scipy.interpolate import interp1d


with open('filters_default.res', 'r') as f:
    lines = f.readlines()


filt = {}

wvl = list()
transmission = list()
name = ''

for i, line in enumerate(lines):
    if line.startswith("#"):
        previous_name = name
        name = line.strip('#').strip()
        if i != 0:
            filt[previous_name] = {}
            filt[previous_name]['wvl'] = np.array(wvl)
            filt[previous_name]['tr'] = np.array(transmission)
        wvl = list()
        transmission = list()
        # previous_name = name
    else:
        data = line.split()
        wvl.append(float(data[0]))
        transmission.append(float(data[1]))

filt[name] = {}
filt[name]['wvl'] = np.array(wvl)
filt[name]['tr'] = np.array(transmission)



filt_aliases = {
 "Buser's U filter (R. Buser)": 'U',
 "Buser's B2 filter (R. Buser)": 'B2',
 "Buser's B3 filter (R. Buser)" : 'B3',
 "Buser's V filter (R. Buser)" : 'V',
 "Johnson's R filter" : 'R_j',
 "Johnson's I filter" : 'I_j',
 "Johnson's J filter": 'J_j',
 "Johnson's K filter": 'K_j',
 "Johnson's L filter": 'L_j',
 'SDSS Camera u Response Function, airmass = 1.3 (June 2001)': 'u',
 'SDSS Camera g Response Function, airmass = 1.3 (June 2001)': 'g',
 'SDSS Camera r Response Function, airmass = 1.3 (June 2001)': 'r',
 'SDSS Camera i Response Function, airmass = 1.3 (June 2001)': 'i',
 'SDSS Camera z Response Function, airmass = 1.3 (June 2001)': 'z',
 '2MASS J filter (total response w/atm)' : 'J_2mass',
 '2MASS H filter (total response w/atm)' : 'H_2mass',
 '2MASS Ks filter (total response w/atm)' : 'Ks_2mass',
 'R Cousins from Bessel (1990)' : 'R_b',
 'I Cousins from Cousins (1980)': 'I_b',
 'WISE W1 (3.4 microns)' : 'W1',
 'WISE W2 (4.6 microns)' : 'W2',
 'Spitzer 3.6 microns' : 'sp36',
 'Spitzer 4.5 microns' : 'sp45'}


FILTERS = filt.copy()

for k in filt.keys():
    # print(k, filt_aliases[k])
    FILTERS[filt_aliases[k]] = FILTERS.pop(k)


def _log(X):
    zeromask = X<=0
    logX = np.empty(X.shape)
    logX[zeromask] = -750.  # the smallest (in magnitude) negative value x for which np.exp(x) returns zero
    logX[~zeromask] = np.log(X[~zeromask])
    return logX


class Filter:
    _available_filters = list(FILTERS.keys())

    def __init__(self, name, wavelengths, transmission):
        self.name = name
        self._Wavelengths = wavelengths
        self._Transmission = transmission
        self._IntegratedTransmission = np.trapz(x=self._Wavelengths, y=self._Transmission)

    @classmethod
    def from_name(cls, name):
        if name not in cls._available_filters:
            raise RuntimeError("Filter {} not availbale".format(name))
        return cls(name, FILTERS[name]['wvl'], FILTERS[name]['tr'])

    @property
    def transmissions(self):
        return self._Transmission

    @property
    def wavelengths(self):
        return self._Wavelengths

    def convolve_spectrum(self, spectrum, **kwargs):
        return self.convolve(spectrum.wavelength, spectrum.flux)


    # Taken from SKIRT codebase, thanks to Sam Verstoken
    def convolve(self, wavelengths, densities, return_grid=False):
        r""" This function calculates and returns the filter-averaged value \f$\left<F_\lambda\right>\f$ for a given
        spectral energy distribution \f$F_\lambda(\lambda)\f$. The calculation depends
        on the filter type. For a photon counter with response curve \f$R(\lambda)\f$,
        \f[ \left<F_\lambda\right> = \frac{ \int\lambda F_\lambda(\lambda)R(\lambda) \,\mathrm{d}\lambda }
            { \int\lambda R(\lambda) \,\mathrm{d}\lambda }. \f]
        For a bolometer with transmission curve \f$T(\lambda)\f$,
        \f[ \left<F_\lambda\right> = \frac{ \int F_\lambda(\lambda)T(\lambda) \,\mathrm{d}\lambda }
            { \int T(\lambda) \,\mathrm{d}\lambda }. \f]

        The quantities \f$F_\lambda(\lambda)\f$ must be expressed per unit of wavelength (and \em not, for example,
        per unit of frequency). The resulting \f$\left<F_\lambda\right>\f$ has the same units as the input distribition.
        \f$F_\lambda(\lambda)\f$ can be expressed in any units (as long as it is per unit of wavelength) and it can
        represent various quantities; for example a flux density, a surface density, or a luminosity density.

        The function accepts two arguments:
        - \em wavelengths: a numpy array specifying the wavelengths \f$\lambda_\ell\f$, in micron, in increasing order,
          on which the spectral energy distribution is sampled. The integration is performed on a wavelength grid that
          combines the grid points given here with the grid points on which the filter response or transmission curve
          is defined.
        - \em densities: a numpy array specifying the spectral energy distribution(s) \f$F_\lambda(\lambda_\ell)\f$
          per unit of wavelength. This can be an array with the same length as \em wavelengths, or a multi-dimensional
          array where the last dimension has the same length as \em wavelengths.
          The returned result will have the shape of \em densities minus the last (or only) dimension.
        """
        # define short names for the involved wavelength grids
        wa = wavelengths
        wb = self._Wavelengths

        # create a combined wavelength grid, restricted to the overlapping interval
        w1 = wa[ (wa>=wb[0]) & (wa<=wb[-1]) ]
        w2 = wb[ (wb>=wa[0]) & (wb<=wa[-1]) ]
        w = np.unique(np.hstack((w1,w2)))
        if len(w) < 2:
            if return_grid: return 0, w
            else: return 0

        # log-log interpolate SED and transmission on the combined wavelength grid
        # (use scipy interpolation function for SED because np.interp does not support broadcasting)
        F = np.exp(interp1d(np.log(wa), _log(densities), copy=False, bounds_error=False, fill_value=0.)(np.log(w)))
        T = np.exp(np.interp(np.log(w), np.log(wb), _log(self._Transmission), left=0., right=0.))

        # perform the integration
        # if self._PhotonCounter:
        #     convolved = np.trapz(x=w, y=w*F*T) / self._IntegratedTransmission
        # else:
        convolved = np.trapz(x=w, y=F*T) / self._IntegratedTransmission
        # Return
        if return_grid: return convolved, w
        else: return convolved

    def plot_transmission(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._Wavelengths, self._Transmission)
        ax.set_xlabel('$\lambda$')
        ax.set_ylabel('$T$')
        ax.set_title(self.name)