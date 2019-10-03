import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from simifucube.write_cube import write_cube
from spectral_cube import SpectralCube

# From Ibarra 2019 article MANGA survey
# k1 = 100
# l1 = 3900
# k2 = 100
# l2 = 10100

# For reproducibility
np.random.seed(14091988)


def sigmoid(l, k, midpoint):
    return 1/(1+np.exp(-1/k * (l-midpoint)))


def shape_function(spectral_axis, l1, l2, k1=100, k2=100, tN=15):
    s1 = partial(sigmoid, k=k1, midpoint=l1)
    s2 = partial(sigmoid, k=-k2, midpoint=l2)
    f = 1 + (tN-1)*(s1(spectral_axis.value) * s2(spectral_axis.value))
    sf = tN/f
    return sf.astype(np.float32)


def ibarra_noise(sp, l1, l2, k1=100, k2=100, tN=15):
    s1 = partial(sigmoid, k=k1, midpoint=l1)
    # Notice the sign here, it's needed for inverting the sigmoid shape
    s2 = partial(sigmoid, k=-k2, midpoint=l2)
    f = 1 + (tN-1)*(s1(sp.spectral_axis.value) * s2(sp.spectral_axis.value))
    sigmas = sp.flux * tN/f
    noise = np.random.normal(loc=0.0, scale=sigmas)
    return noise


def plot_sigmas(cube, l1, l2, k1, k2, tN):
    sf = shape_function(cube.spectral_axis, l1, l2, k1, k2, tN)
    plt.plot(cube.spectral_axis, sf)

# def plot_noise(sp, noise, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     ax.plot(sp.spectral_axis, noise/sp.flux)
#     ax.set_ylabel('$N/F_0$')
#     ax.set_xlabel('$\AA$')


def plot_noise(noise_cube, x=None, y=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if x is None:
        x = noise_cube.shape[1]//2
    if y is None:
        y = noise_cube.shape[2]//2
    ax.plot(noise_cube.spectral_axis, noise_cube._data[:,x,y])
    ax.set_ylabel('$N$')
    ax.set_xlabel('$\AA$')


def compute_sigmas(cube, sigma_lim, l1, l2, k1=1000, k2=1000, tN=15):
    sf = shape_function(cube.spectral_axis, l1, l2, k1, k2, tN)
    # cube.allow_huge_operations=True
    sigmas = (np.ones(cube.shape, dtype=np.float32) * sigma_lim * sf[:, np.newaxis, np.newaxis])
    # sigmas = sigma_lim * sf
    # sigma_cube = np.output_name.
    return sigmas


def get_variance_cube(cube, sigmas):
    variances = sigmas**2
    variance_cube = SpectralCube(data=variances*(cube.unit)**2, wcs=cube.wcs, header=cube.header)
    return variance_cube


def get_error_cube(cube, sigmas):
    noise = np.random.normal(loc=0.0, scale=sigmas).astype(np.float32)
    noise_cube = SpectralCube(data=noise*cube.unit, wcs=cube.wcs, header=cube.header)
    return noise_cube


def compute_errors(cube, sigma_lim, l1, l2, k1=1000, k2=1000, tN=15):
    sigmas = compute_sigmas(cube, sigma_lim, l1, l2, k1, k2, tN)
    noise_cube = get_error_cube(cube, sigmas)
    variance_cube = get_variance_cube(cube, sigmas)
    return noise_cube, variance_cube


def main(cube_filename, sigma_lim, l1, l2, output_name, k1=1000, k2=1000, tN=15, overwrite=False):
    cube = SpectralCube.read(cube_filename)
    noise_cube, variance_cube = compute_errors(cube, sigma_lim, l1, l2, k1, k2, tN)
    noisy_cube = cube + noise_cube
    write_cube(noisy_cube, variance_cube, output_name, overwrite)


MUSE_SP_AX_RES = 1.25  ## Angstrom

if __name__ == '__main__':
    cube_name = 'cube_produced/published/cube_69p2_0227_r3_b80_err.fits'
    cube = SpectralCube.read(cube_name)
    filename = 'prova.fits'
    k = 1000
    # limits = 6000, 8500
    limits = -100000, 30000  # No limits
    tN = 10
    # from Figure 21 Law et al.(2016)
    # sigma_lim = 4 * 1000 /(MUSE_SP_AX_RES * 5 ) = 640  # BAD!  # 1e-20 erg s-1 cm-2
    # From median of the variance in a real datacube:
    sigma_lim = 4
    print("sigma_lim =", sigma_lim)
    plot_sigmas(cube, *limits, k, k, tN)
    sigmas = compute_sigmas(cube, sigma_lim, *limits, k, k, tN)
    noise_cube = get_error_cube(cube, sigmas)
    variance_cube = get_variance_cube(cube, sigmas)
    noisy_cube = cube + noise_cube
    write_cube(noisy_cube, variance_cube, filename, overwrite=True)
    plt.show()
