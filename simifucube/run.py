#!/usr/bin/env python3

import warnings

import numpy as np
import pynbody
from astropy.io.fits.verify import VerifyWarning


from spectral_cube import SpectralCube

from simifucube.generate_spectra import SnapSpectra, muse_rebin
from simifucube.cube_generator import CubeGenerator
from simifucube.write_cube import write_cube
from simifucube.der_snr import DER_SNR

config = dict(
use_template_star=True,
do_preprocessing=False,

num_threads=1,

size_cuboid = 10,

bins = 20,
do_spectral_rebinning=True,

output_LOS_sum=False,
)

# I/O
# snap_name = 'toy_snap_distribution.snap'
snap_name = 'toy_snap_d4.0_M1e4'
# snap_name = 'toy_snap_multi_Vm60p80'
pynbody.config['sph']['smooth-particles'] = 2
pynbody.config['sph']['tree-leafsize'] = 1
out_name='toycube_distrib' #8particlesVm60p80NONOISE'
pickle_name = 'toy_distribution.pkl'


snsp = SnapSpectra(snap_name, size_cuboid=config['size_cuboid'], do_preprocessing=config['do_preprocessing'])
snsp.generate_spectra(save_as=pickle_name, use_template_star=config['use_template_star'])
# print(snsp._preprocessed_snap['smooth'])
# print(snsp._preprocessed_snap['mass'])
# snsp.generate_spectra_from_pickle(pickle_name)

cg = CubeGenerator(snsp, bins=config['bins'])
if config['output_LOS_sum']:
    cg.sum_on_line_of_sigth()
    cube = cg.create_cube()
    cube.write(out_name+'pix{}_sum.fits'.format(config['bins']), overwrite=True)


im = cg.sph_projection_direct(num_threads=config['num_threads'])

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


if config['do_spectral_rebinning']:
    muse_cube = muse_rebin(snsp.last_valid_freq, cube_sph)
else:
    muse_cube = cube_sph

print('Computing STAT HDU')
s = np.shape(muse_cube._data)
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
warnings.simplefilter('ignore', category=VerifyWarning)
write_cube(muse_cube, variance_cube=variance_cube, filename=out_name+'_r{}pix{}.fits'.format(config['size_cuboid'],config['bins']), meta=config, overwrite=True)
