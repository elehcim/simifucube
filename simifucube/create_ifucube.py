#!/usr/bin/env python3

import warnings
from configparser import ConfigParser
from argparse import ArgumentParser

import numpy as np
import pynbody
from astropy.io.fits.verify import VerifyWarning

from spectral_cube import SpectralCube

from simifucube.generate_spectra import SnapSpectra, muse_rebin
from simifucube.cube_generator import CubeGenerator
from simifucube.write_cube import write_cube
from simifucube.util.der_snr import DER_SNR



# config = dict(
# use_template_star=True,
# do_preprocessing=False,

# num_threads=1,

# size_cuboid = 10,

# bins = 10,
# do_spectral_rebinning=True,

# output_LOS_sum=False,
# output_SPH_projection=True,

# do_noise=False,
# STAT_HDU=True,

# # We defined the contamination range (the dispersion) in a way that the residuals
# # between the constructed spectrum and the original spectrum of
# # the library to be less than ~ 0.073.
# sigma=0.07,

# # I/O
# # snap_name = 'toy_snap_cloud_n1000_r1d4Vp20m-500_s100.snap',
# snap_name = 'toySnapCloudCubeN1000r1d4Vp20m500s100.snap',
# out_name='toycubedistribCubeNONOISE', #8particlesVm60p80NONOISE'
# pickle_name = 'toy_distribution.pkl',
# )

# I/O
# snap_name = 'toy_snap_distribution.snap'
# snap_name = 'toy_snap_d4.0_M1e4_sm1'
# snap_name = 'toy_snap_d3_M1e4_sm1'

# snap_name = 'toy_snap_multi_Vm60p80'
# pynbody.config['sph']['smooth-particles'] = 2
# pynbody.config['sph']['tree-leafsize'] = 1

def generate_cube(config):
    out_name = config['out_name']
    bins = config.getint('bins')
    size_cuboid = config.getint('size_cuboid')

    snsp = SnapSpectra(config['snap_name'],
                       size_cuboid=size_cuboid,
                       do_preprocessing=config.getboolean('do_preprocessing'))

    snsp.generate_spectra(save_as=config['pickle_name'],
                          use_template_star=config.getboolean('use_template_star'))
    # print(snsp._preprocessed_snap['smooth'])
    # print(snsp._preprocessed_snap['mass'])
    # snsp.generate_spectra_from_pickle(pickle_name)

    cg = CubeGenerator(snsp, bins=bins)

    if config.getboolean('output_LOS_sum'):
        im_los = cg.sum_on_line_of_sigth()
        print('datacube max flux', np.max(cg.datacube))

        cube = cg.create_spectral_cube()
        sum_outname = out_name + 'pix{}_sum.fits'.format(bins)
        print(f'Writing cube {sum_outname}')
        cube.write(sum_outname, overwrite=True)

    if config.getboolean('output_SPH_projection'):
        im = cg.sph_projection_direct(num_threads=config.getint('num_threads'))
        print('datacube max flux', np.max(cg.datacube))

    # Do noise:
    if config.getboolean('do_noise'):
        print('Computing noise')
        # We defined the contamination range (the dispersion ) in a way that the residuals
        # between the constructed spectrum and the original spectrum of
        # the library to be less than ~ 0.073.

        sigma = config.getfloat('sigma')
        noise = np.random.normal(loc=0.0, scale=sigma*cg.datacube)
        print('Adding noise to signal')
        cg.datacube += noise


    # Creating actual cube
    cube_sph = cg.create_spectral_cube()


    if config.getboolean('do_spectral_rebinning'):
        muse_cube = muse_rebin(snsp.last_valid_freq, cube_sph)
    else:
        muse_cube = cube_sph

    if config.getboolean('STAT_HDU'):
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
               filename=out_name+'r{}pix{}.fits'.format(size_cuboid,bins),
               meta=dict(config),
               overwrite=True)

def main(cli=None):
    parser = ArgumentParser()
    parser.add_argument(dest='configfile', help="Path to the configuration file")
    args = parser.parse_args(cli)

    configurator = ConfigParser()
    configurator.read(args.configfile)
    config = configurator['IFU']

    if configurator['general'].getboolean('pipe'):
        config['snap_name'] = generate_outname_from_config(configurator['toycube'])

    print(dict(config))
    generate_cube(config)

if __name__ == '__main__':
    main()
