import numpy as np
import tqdm
import pynbody
from astropy.wcs import WCS
import time
from simifucube.examples.create_noisy_cube import get_header
from spectral_cube import SpectralCube, LazyMask
from scipy.stats import binned_statistic_2d
from simifucube.render_cube import render_cube

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
        cube = SpectralCube(data=self.datacube.astype(np.float32) * spectra1d.flux.unit,
                            wcs=wcs,
                            mask=mask,
                            header=header)
        return cube
