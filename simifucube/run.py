#!/usr/bin/env python3

import warnings

import numpy as np
import pynbody
from astropy.io.fits.verify import VerifyWarning

from spectral_cube import SpectralCube

from simifucube.generate_spectra import SnapSpectra, muse_rebin
from simifucube.cube_generator import CubeGenerator
from simifucube.write_cube import write_cube
from simifucube.util.der_snr import DER_SNR

config = dict(
use_template_star=True,
do_preprocessing=False,

num_threads=1,

size_cuboid = 10,

bins = 100,
do_spectral_rebinning=True,

output_LOS_sum=False,
output_SPH_projection=True,

do_noise=False,
STAT_HDU=True,

# We defined the contamination range (the dispersion ) in a way that the residuals
# between the constructed spectrum and the original spectrum of
# the library to be less than ~ 0.073.
sigma=0.07,

# I/O
snap_name = 'toy_snap_d3_M1e4_sm1',
out_name='toycube_distrib', #8particlesVm60p80NONOISE'
pickle_name = 'toy_distribution.pkl',
)

# I/O
# snap_name = 'toy_snap_distribution.snap'
# snap_name = 'toy_snap_d4.0_M1e4_sm1'
# snap_name = 'toy_snap_d3_M1e4_sm1'

# snap_name = 'toy_snap_multi_Vm60p80'
pynbody.config['sph']['smooth-particles'] = 2
pynbody.config['sph']['tree-leafsize'] = 1

out_name = config['out_name']


snsp = SnapSpectra(config['snap_name'],
                   size_cuboid=config['size_cuboid'],
                   do_preprocessing=config['do_preprocessing'])

snsp.generate_spectra(save_as=config['pickle_name'],
                      use_template_star=config['use_template_star'])
# print(snsp._preprocessed_snap['smooth'])
# print(snsp._preprocessed_snap['mass'])
# snsp.generate_spectra_from_pickle(pickle_name)

cg = CubeGenerator(snsp, bins=config['bins'])

if config['output_LOS_sum']:
    im_los = cg.sum_on_line_of_sigth()
    print('datacube max flux', np.max(cg.datacube))

    cube = cg.create_spectral_cube()
    sum_outname = out_name + 'pix{}_sum.fits'.format(config['bins'])
    print(f'Writing cube {sum_outname}')
    cube.write(sum_outname, overwrite=True)

if config['output_SPH_projection']:
    im = cg.sph_projection_direct(num_threads=config['num_threads'])
    print('datacube max flux', np.max(cg.datacube))

# Do noise:
if config['do_noise']:
    print('Computing noise')
    # We defined the contamination range (the dispersion ) in a way that the residuals
    # between the constructed spectrum and the original spectrum of
    # the library to be less than ~ 0.073.

    sigma = config['sigma']
    noise = np.random.normal(loc=0.0, scale=sigma*cg.datacube)
    print('Adding noise to signal')
    cg.datacube += noise


# Creating actual cube
cube_sph = cg.create_spectral_cube()


if config['do_spectral_rebinning']:
    muse_cube = muse_rebin(snsp.last_valid_freq, cube_sph)
else:
    muse_cube = cube_sph

if config['STAT_HDU']:
    print('Computing STAT HDU')
    s = np.shape(muse_cube._data)
    print('shape muse_cube:', s)
    just_spectra  = np.reshape(muse_cube._data,[s[0],s[1]*s[2]])
    n_spectra = just_spectra.shape[1]
    print("Estimating the error spectra with the DER_SNR algorithm")
    error_spectra = np.zeros(just_spectra.shape)
    for i in range(n_spectra):
        error_spectra[:,i] = DER_SNR(just_spectra[:,i])
        # np.abs(np.nanmedian(np.sqrt(error_spectra),axis=0))
    stat_data = np.reshape(error_spectra, muse_cube.shape).astype(np.float32)
    variance_cube = SpectralCube(data=stat_data**2 * (muse_cube.unit)**2,
                                 wcs=muse_cube.wcs,
                                 header=muse_cube.header)
else:
    variance_cube = None

# muse_cube.write(out_name+'pix{}.fits'.format(bins), overwrite=True)

# this is for ignoring the HIERARCH keywords warning
warnings.simplefilter('ignore', category=VerifyWarning)
write_cube(muse_cube,
           variance_cube=variance_cube,
           filename=out_name+'_r{}pix{}.fits'.format(config['size_cuboid'],config['bins']),
           meta=config,
           overwrite=True)
