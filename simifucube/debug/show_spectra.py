from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
# cube = SpectralCube.read('cube_69p2_0227_r3_b80.fits')
cube = SpectralCube.read('cube_produced/published/20190131/cube_69p2_0227_r3_b80_sigma4.fits', hdu='STAT')
# cube = SpectralCube.read('dataset/VCC1833/big.fits', hdu=2)
# cube = SpectralCube.read('smcube.fits')


# pix_x = 40
# pix_y = 33
# cube[:, pix_x, pix_y].quicklook()
cube[:, 40, 33].quicklook()
cube[:, 41, 33].quicklook()

plt.show()