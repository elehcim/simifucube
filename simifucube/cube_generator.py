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

import time

import numpy as np
import pynbody
import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from spectral_cube import SpectralCube, LazyMask

from simifucube.render_cube import render_cube

HEADER_SOURCE = '/home/michele/sim/IFU_images/dataset/NGC1427A/ADP.2016-06-14T15:15:54.554.fits'


def get_header(data_source=HEADER_SOURCE):
    print('Getting template header from', data_source)
    with fits.open(data_source) as hdulist:
        _header = hdulist[1].header
    return _header

class MySpectralCube(SpectralCube):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def spectral_smooth_variable_width(self, convolve, kernel, var,
                                       verbose=0,
                                       use_memmap=True,
                                       num_cores=None,
                                       **kwargs):
        """
        Smooth the cube along the spectral dimension with a variable smoothing kernel

        Note that the mask is left unchanged in this operation.

        Parameters
        ----------
        convolve : function(spectrum, kernel, var)
            Operates on the spectra the spectrum values,
        kernel : np.array
            A 1D kernel array
        verbose : int
            Verbosity level to pass to joblib
        use_memmap : bool
            If specified, a memory mapped temporary file on disk will be
            written to rather than storing the intermediate spectra in memory.
        num_cores : int or None
            The number of cores to use if running in parallel
        kwargs : dict
            Passed to the convolve function
        """
        def my_convolve(y, kernel=kernel, **kwargs):
            return convolve(self.spectral_axis.view(np.ndarray), y, kernel=kernel, var=var)

        # convolve = kwargs.get('my_convolve', convolve)
        return self.apply_function_parallel_spectral(my_convolve,
                                                     kernel=kernel,
                                                     normalize_kernel=True,
                                                     num_cores=num_cores,
                                                     use_memmap=use_memmap,
                                                     verbose=verbose,
                                                     **kwargs)



class CubeGenerator:
    def __init__(self, snap_spectra, bins):
        self.snap_spectra = snap_spectra
        self.bins = bins
        self.star_snap = self.snap_spectra._preprocessed_snap.s
        self._spectra_assigned = False
        self._spectra_channels_assigned = False

    def _create_bins(self):
        n_spectra = binned_statistic_2d(self.snap_spectra.x, self.snap_spectra.y,
                            np.ones_like(self.snap_spectra.spectrum.flux.transpose(1,0)),
                            statistic='count',
                            bins=self.bins,
                            expand_binnumbers=True)
        center_x = (n_spectra.x_edge[1:] + n_spectra.x_edge[:-1])/2
        center_y = (n_spectra.y_edge[1:] + n_spectra.y_edge[:-1])/2
        return center_x, center_y, n_spectra

    def _assign_spectra_channel_to_star_particles(self):
        print('Assigning spectra channels to star_particles')
        for i in tqdm.tqdm(self.n_channels):
            self.star_snap['sp_ch{:06d}'.format(i)] = self.snap_spectra.spectrum.flux[:,i]
        self._spectra_channels_assigned = True

    def sph_projection_direct(self, num_threads=None):
        t0 = time.time()
        print('removing boxsize')
        if 'boxsize' in self.star_snap.properties:
            del self.star_snap.properties['boxsize']
        # print(self.star_snap.properties)
        print('Doing projection')
        print('flux  ', np.nan_to_num(self.snap_spectra.spectrum.flux.view(np.ndarray), copy=True))
        print('pos   ', self.star_snap['pos'])
        print('smooth', self.star_snap['smooth'])
        print('mass  ', self.star_snap['mass'])
        print('rho   ', self.star_snap['rho'])
        im_cube = render_cube(
                self.star_snap,
                qty=np.nan_to_num(self.snap_spectra.spectrum.flux.view(np.ndarray), copy=True),
                nx=self.bins,
                x2=self.snap_spectra._size_cuboid,
                x1=-self.snap_spectra._size_cuboid,
                kernel=pynbody.sph.Kernel2D(),
                num_threads=num_threads)

        print('SPH projection done in {:.3f} s'.format(time.time()-t0))

        self.datacube = im_cube.transpose(2, 0, 1)
        return self.datacube

    def sph_projection(self):
        if not self._spectra_assigned:
            self._assign_spectra_channel_to_star_particles()
        im_list = list()
        with self.star_snap.immediate_mode:
            for i in tqdm.tqdm(self.n_channels):
                im_list.append(pynbody.sph.render_image(
                    self.star_snap,
                    qty='sp_ch{:06d}'.format(i),
                    nx=self.bins,
                    x2=self.snap_spectra._size_cuboid,
                    kernel=pynbody.sph.Kernel2D(),
                    threaded=False).view(np.ndarray))

        self.datacube = np.stack(im_list)
        return self.datacube

    def sum_on_line_of_sigth(self):
        flux = np.nan_to_num(self.snap_spectra.spectrum.flux.view(np.ndarray), copy=True)
        self.binned_stat = binned_statistic_2d(self.snap_spectra.pos[0], self.snap_spectra.pos[1],
                                flux.transpose(1,0),
                                statistic='sum',
                                bins=self.bins,
                                expand_binnumbers=True)

        self.datacube = self.binned_stat.statistic.astype(np.float32).transpose(0,2,1)

    @property
    def n_channels(self):
        # return range(3412, 8801)
        return self.snap_spectra.spectrum.flux.shape[1]

    @property
    def n_stars(self):
        return self.snap_spectra.spectrum.flux.shape[0]

    def create_spectral_cube(self, mask=None, wcs=None, header=None):
        if not hasattr(self, 'datacube'):
            raise RuntimeError('First run an aggregator function')
        spectra1d = self.snap_spectra.spectrum
        if header is None:
            _header = get_header()
            wvl = spectra1d.spectral_axis.value
            wvl_diff = np.diff(wvl)
            # Check if first diff is actually the same everywhere
            if not np.allclose(wvl_diff[0] * np.ones_like(wvl[:-1]), wvl_diff, rtol=1e-05, atol=1e-08):
                raise RuntimeError('Spectral axis has varying resolution')
            _header['CD3_3'] = wvl_diff[0]
            _header['CRVAL3'] = wvl[0]
            _header['CRPIX1'] = _header['CRPIX2'] = self.bins/2
            header = _header

        if wcs is None:
            wcs = WCS(header)

        if mask is None:
            mask = LazyMask(np.isfinite, data=self.datacube, wcs=wcs)

        print('Creating cube...')
        cube = MySpectralCube(data=self.datacube.astype(np.float32) * spectra1d.flux.unit,
                              wcs=wcs,
                              mask=mask,
                              header=header)
        return cube
