import os
import sys
import warnings

import numpy as np
import pynbody
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.io import fits
import astropy.units as u
from astropy.table import Table

from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from spectral_cube import SpectralCube, LazyMask

import pickle

from generate_spectra import MUSE_LIMITS, SnapSpectra, muse_rebin
from cube_generator import CubeGenerator
from render_cube import render_cube
from write_cube import write_cube
from der_snr import DER_SNR

MORIA_PATH='/mnt/data/MoRIA/M1-10_Verbeke2017/M10sim41001'
SIM_PATH = '/home/michele/sim/MySimulations/ng'
sim = 62002
peri = 200
isnap = 227

_do_preprocessing=True
use_template_star=False
num_threads=1

moria=False
rit38=False
toy = False
toy_multi=True

if moria:
    # snap_name = "/mnt/data/MoRIA/M1-10_Verbeke2017/M01sim14001/snapshot_0000"
    snap_name = "/mnt/data/MoRIA/M1-10_Verbeke2017/M09sim69001/snapshot_0036"
    # snap_name = os.path.join(MORIA_PATH, 'snapshot_0036')
    out_name = 'M09sn36_noisy'
elif rit38:
    snap_name = os.path.join("/2disk/mmastrop/M1-10_Verbeke2017/M10sim41001/snapshot_0036")
elif toy:
    snap_name = 'toy_snap_d4.0'
    pynbody.config['sph']['smooth-particles'] = 2
    pynbody.config['sph']['tree-leafsize'] = 1
    out_name='toycubepix20METm1.5AGE3Vm20p500d4.0_noise'
    _do_preprocessing=False
    use_template_star=True
elif toy_multi:
    snap_name = 'toy_snap_multi'
    snap_name = 'toy_snap_multi_Vm60p80'
    pynbody.config['sph']['smooth-particles'] = 2
    pynbody.config['sph']['tree-leafsize'] = 1
    out_name='toycube8particlesVm60p80NONOISE'
    _do_preprocessing=False
    use_template_star=False
else:
    snap_name = os.path.join(SIM_PATH, "mb.{}_p{}_a800_r600/out/snapshot_{:04d}".format(sim, peri, isnap))
    out_name = '{}p{}_sn{}'.format(62002, peri, isnap)

size_cuboid = 10
bins = 20
do_spectral_rebinning=True


pickle_name = 'test_pickle.pkl'

snsp = SnapSpectra(snap_name, size_cuboid=size_cuboid, do_preprocessing=_do_preprocessing)
snsp.generate_spectra(save_as=pickle_name, use_template_star=use_template_star)
print(snsp._preprocessed_snap['smooth'])
print(snsp._preprocessed_snap['mass'])
# snsp.generate_spectra_from_pickle(pickle_name)

cg = CubeGenerator(snsp, bins=bins)
cg.sum_on_line_of_sigth()
cube = cg.create_cube()
cube.write(out_name+'pix{}_sum.fits'.format(bins), overwrite=True)

im = cg.sph_projection_direct(num_threads=num_threads)

# Do noise:
print('Computing noise')
# We defined the contamination range (the dispersion ) in a way that the residuals
# between the constructed spectrum and the original spectrum of
# the library to be less than ~ 0.073.

sigma = 0.07
noise = np.random.normal(loc=0.0, scale=sigma*cg.datacube)
print('Adding noise to signal')
cg.datacube += noise


cube_sph = cg.create_cube()


if do_spectral_rebinning:
    muse_cube = muse_rebin(snsp.last_valid_freq, cube_sph)
else:
    muse_cube = cube_sph

print('Computing STAT HDU')
s     = np.shape(muse_cube._data)
print(s)
just_spectra  = np.reshape(muse_cube._data,[s[0],s[1]*s[2]])
n_spectra = just_spectra.shape[1]
print("Estimating the error spectra with the DER_SNR algorithm")
error_spectra = np.zeros(just_spectra.shape)
for i in range(n_spectra):
    error_spectra[:,i] = DER_SNR(just_spectra[:,i])
    # np.abs(np.nanmedian(np.sqrt(error_spectra),axis=0))
stat_data = np.reshape(error_spectra, muse_cube.shape).astype(np.float32)
variance_cube = SpectralCube(data=stat_data**2 * (muse_cube.unit)**2, wcs=muse_cube.wcs, header=muse_cube.header)

# muse_cube.write(out_name+'pix{}.fits'.format(bins), overwrite=True)

write_cube(muse_cube, variance_cube=variance_cube, filename=out_name+'_r{}pix{}.fits'.format(size_cuboid,bins), overwrite=True)