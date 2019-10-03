# coding: utf-8
from spectral_cube import SpectralCube
from astropy.io import fits
import matplotlib.pyplot as plt
cube = SpectralCube.read('cube_69p2_0227_r3_b80_single_spectra.fits.gz')
cube
model_name = 'library/EMILES_BASTI_BASE_CH_FITS/Ech1.30Zm1.49T03.0000_iTp0.00_baseFe.fits'
hdul = fits.open(model_name)
hdul[0]
hdul[0].data
model_flux = hdul[0].data[:10000]
model_flux
cube_flux = cube[:, 40, 40]
cube_flux
cube_flux.shape
res = cube_flux / model_flux
res
plt.plot(res)
plt.show()
plt.plot(cube_flux)
plt.plot(model_flux*64157.344)
plt.show()
