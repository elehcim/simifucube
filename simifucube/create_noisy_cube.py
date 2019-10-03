import os
import sys
import warnings

import numpy as np
import pynbody
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from scipy.stats import binned_statistic_2d
from spectral_cube import SpectralCube, LazyMask

from simifucube.compute_error import compute_errors, get_error_cube
from simifucube.write_cube import contract_name, write_cube
from simifucube.generate_spectra import spectra_from_snap, spectra_from_pickle, MUSE_LIMITS


warnings.simplefilter(action='ignore', category=FutureWarning)

SIM_PATH = '/home/michele/sim/MySimulations/ng'

HEADER_SOURCE = '/home/michele/sim/IFU_images/dataset/NGC1427A/ADP.2016-06-14T15:15:54.554.fits'

def get_header(data_source=HEADER_SOURCE):
    print('Getting template header from', data_source)
    with fits.open(data_source) as hdulist:
        _header = hdulist[1].header
    return _header


def create_cube(spectra1d, x, y, bins, mask=None, wcs=None, header=None):
    print('Aggregating spectra in {} bins...'.format(bins))
    if header is None:
        _header = get_header()
        wvl = spectra1d.spectral_axis.value
        wvl_diff = np.diff(wvl)
        # Check if first diff is actually the same everywhere
        if not np.allclose(wvl_diff[0] * np.ones_like(wvl[:-1]), wvl_diff, rtol=1e-05, atol=1e-08):
            raise RuntimeError('Spectral axis has varying resolution')
        _header['CD3_3'] = wvl_diff[0]
        _header['CRVAL3'] = wvl[0]
        _header['CRPIX1'] = _header['CRPIX2'] = bins/2
        header = _header

    if wcs is None:
        wcs = WCS(header)

    a = binned_statistic_2d(x,y, spectra1d.flux.transpose(1,0),
                            statistic='sum',
                            bins=bins,
                            expand_binnumbers=True)
    if mask is None:
        mask = LazyMask(np.isfinite, data=a.statistic, wcs=wcs)

    print('Creating cube...')
    cube = SpectralCube(data=a.statistic.astype(np.float32) * spectra1d.flux.unit,
                        wcs=wcs,
                        mask=mask,
                        header=header)

    return a, cube


def get_spectra(snap_name, save_pickle, size_cuboid, force=False, use_fix_val=False, doppler_shift=True):
    if not force and os.path.isfile(save_pickle):
        print('Reading from file: {}'.format(save_pickle))
        out = spectra_from_pickle(save_pickle)
    else:
        print('Getting spectra from ', snap_name)
        out = spectra_from_snap(snap_name,
                                save_as=save_pickle,
                                size_cuboid=size_cuboid,
                                doppler_shift=doppler_shift,
                                use_fix_val=use_fix_val)
    spectra1d, (x,y), last_valid_freq = out
    return spectra1d, (x,y), last_valid_freq


def get_spatial_kernelwidth(sigma_psf, fov, npix):  ## arcsec
    arcsec_per_pix = fov/npix
    print("Plate scale: {:.2f} arcsec/pix".format(arcsec_per_pix))
    kernelwidth = sigma_psf/arcsec_per_pix  # pix
    # print("Sigma_psf: {:.2f} pix".format(kernelwidth))
    return kernelwidth


def sigma_from_fwhm(fwhm):
    return fwhm/np.sqrt(8 * np.log(2))  # 2.35

# from http://muse.univ-lyon1.fr/spip.php?article102
# Resolving Power:
# 1770 @ 480 nm
# 2850 @ 750 nm
# 3590 @ 930 nm
# so, I take the mid one: delta_lambda = lambda/R = 7500/2850 = 2.632 Angstrom
# See here the graph: https://www.eso.org/sci/facilities/paranal/instruments/muse/inst.html
DELTA_LAMBDA_MUSE = 2.632  ## Angstrom
MUSE_SP_AX_RES = 1.25  ## Angstrom

def get_spectral_kernelwidth(spectral_axis_resolution, delta_lambda=DELTA_LAMBDA_MUSE):
    """ delta_lambda = lambda/R """
    kernelwidth = sigma_from_fwhm(delta_lambda)
    # print('sigma_spectral: {:.2f} Angstrom'.format(kernelwidth))
    # print('sigma_spectral: {:.2f} pix'.format(kernelwidth/spectral_axis_resolution))
    return kernelwidth/spectral_axis_resolution  ## sp_pix



def create_single_spectra_cube(bins, mask=None, header=None, wcs=None):
    from generate_spectra import STD_MET, STD_AGE, to_spectrum1d
    from spectra import Spectrum
    mass = 8000  # Msol
    print('Creating cube from a particle with age{}, met={}, mass={} Msol'.format(STD_AGE, STD_MET, mass))

    sp = Spectrum.from_met_age(STD_MET, STD_AGE)
    L_sol = 3.839e33  # erg s-1
    kpc_in_cm = 3.086e+21  # cm
    pos = np.array([0.0,0.0,0.0])
    z_dist = 20000  # kpc
    dist_sq = np.linalg.norm(pos - np.array([0, 0, z_dist]))**2
    sp.flux *= mass * L_sol / (4 * np.pi * dist_sq * kpc_in_cm**2)
    sp_list = [sp] * bins**2
    spectra1d = to_spectrum1d(sp_list)

    if header is None:
        _header = get_header()
        wvl = spectra1d.spectral_axis.value
        wvl_diff = np.diff(wvl)
        # Check if first diff is actually the same everywhere
        if not np.allclose(wvl_diff[0] * np.ones_like(wvl[:-1]), wvl_diff, rtol=1e-05, atol=1e-08):
            raise RuntimeError('Spectral axis has varying resolution')
        _header['CD3_3'] = wvl_diff[0]
        _header['CRVAL3'] = wvl[0]
        _header['CRPIX1'] = _header['CRPIX2'] = bins/2
        header = _header

    if wcs is None:
        wcs = WCS(header)

    data = spectra1d.data.reshape(bins, bins, spectra1d.shape[1]).astype(np.float32).transpose(2, 0, 1)

    if mask is None:
        mask = LazyMask(np.isfinite, data=data, wcs=wcs)
    cube = SpectralCube(data=data * spectra1d.flux.unit, wcs=wcs, mask=mask, header=header)
    print("Writing output...")
    output_name = contract_name("ssp_cube_b{}{}.fits".format(bins, '_nods' if not doppler_shift else ''))
    write_cube(final_cube, variance_cube, output_name, overwrite_output)
    sys.exit(0)

def main(**kwargs):
    print('Inputs')

    print("Computing cube spectra...")

    print("Spatial smoothing...")

    print("Spectral smoothing...")

    print("Computing cube errors...")

    print("Writing output...")


if __name__ == '__main__':
    print('inputs')
    import matplotlib.pyplot as plt
    sim = 69002
    peri = 200
    isnap = 227

    fix_star_met_age = False

    num_cores = None

    size_cuboid = 3
    bins = 80

    doppler_shift = True

    force = False
    overwrite_output = True

    add_noise = True
    use_der_snr = True

    sigma_inst = .1  # flux unit
    PSF = 1.21  # arcsec (FWHM)  why? source?
    fov = 63.2  # arcsec

    k = 1000
    limits = -100000, 30000  # No limits
    # limits = 5000, 8000
    tN = 10

    do_spectral_smoothing = True
    do_spatial_smoothing = True
    do_spectral_rebinning = True


    # cube = create_single_spectra_cube(bins=bins)



    snap_name = os.path.join(SIM_PATH, "mb.{}_p{}_a800_r600/out/snapshot_{:04d}".format(sim, peri, isnap))
    print('Star mass: {:.2g} Msol'.format(pynbody.load(snap_name).s['mass'].sum().in_units('Msol')))

    save_pickle = contract_name("sp", sim, isnap, peri, size_cuboid, ext='pkl', fix=fix_star_met_age, doppler_shift=doppler_shift)

    print("Computing cube spectra...")
    spectra1d, (x,y), last_valid_freq = get_spectra(snap_name=snap_name,
                                                    save_pickle=save_pickle,
                                                    size_cuboid=size_cuboid,
                                                    force=force,
                                                    doppler_shift=doppler_shift,
                                                    use_fix_val=fix_star_met_age)

    print("Got spectra of shape {}, last_valid_freq={:.2f}".format(spectra1d.shape, last_valid_freq))

    stats, cube = create_cube(spectra1d, x, y, bins)
    # cube.write('signal_c.fits', overwrite=overwrite_output)


    if do_spectral_smoothing:
        print("Spectral smoothing...")
        spectral_kernelwidth = get_spectral_kernelwidth(spectral_axis_resolution=cube.header['CDELT3'])
        print('sigma_spectral: {:.2f} pix'.format(spectral_kernelwidth))
        spectral_kernel = Gaussian1DKernel(spectral_kernelwidth)
        spec_sm = cube.spectral_smooth(spectral_kernel, num_cores=num_cores)
    else:
        spec_sm = cube

    if do_spatial_smoothing:
        print("Spatial smoothing...")
        spatial_kernelwidth = get_spatial_kernelwidth(sigma_from_fwhm(PSF), fov, bins)
        print('sigma_spatial:  {:.2f} pix'.format(spatial_kernelwidth))
        spatial_kernel = Gaussian2DKernel(spatial_kernelwidth)
        smoothed_cube = spec_sm.spatial_smooth(spatial_kernel, num_cores=num_cores)
        smoothed_cube._data = smoothed_cube._data.astype(np.float32)
    else:
        smoothed_cube = spec_sm

    if do_spectral_rebinning:
        print("Adapting spectra to MUSE range and spectral resolution...")
        if last_valid_freq > MUSE_LIMITS['stop']:
            last_valid_freq = MUSE_LIMITS['stop']

        new_bins = np.arange(MUSE_LIMITS['start'], last_valid_freq, MUSE_LIMITS['step']) * u.AA
        print(new_bins)
        muse_cube = smoothed_cube.spectral_interpolate(new_bins)

    else:
        muse_cube = smoothed_cube

    if add_noise:
        if use_der_snr:
            from der_snr import DER_SNR
            s     = np.shape(muse_cube)
            print(s)
            just_spectra  = np.reshape(muse_cube._data,[s[0],s[1]*s[2]])
            print(just_spectra)
            n_spectra = just_spectra.shape[1]
            print("Computing error spectra with DER_SNR algorithm")
            error_spectra = np.zeros(just_spectra.shape)
            for i in range(n_spectra):
                error_spectra[:,i] = DER_SNR(just_spectra[:,i])
                # np.abs(np.nanmedian(np.sqrt(error_spectra),axis=0))
            sigma_data = np.reshape(error_spectra, muse_cube.shape).astype(np.float32)
            noise_cube = get_error_cube(muse_cube, sigma_data)
            variance_cube = SpectralCube(data=sigma_data**2 * (muse_cube.unit)**2, wcs=muse_cube.wcs, header=muse_cube.header)
            final_cube = muse_cube + noise_cube

        else:

            print("\nComputing cube errors...")
            noise_cube, variance_cube = compute_errors(cube=muse_cube,
                                                       sigma_lim=sigma_inst,
                                                       l1=limits[0], l2=limits[1],
                                                       k1=k, k2=k,
                                                       tN=tN)


            print('Adding noise...')
            final_cube = muse_cube + noise_cube
    else:
        final_cube = muse_cube
        variance_cube = None


    print("Writing output...")
    output_name = contract_name("cube", sim, isnap, peri, size_cuboid, ext='fits', bins=bins, fix=fix_star_met_age, doppler_shift=doppler_shift)
    write_cube(final_cube, variance_cube, output_name, overwrite_output)
