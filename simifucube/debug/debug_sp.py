import numpy as np
import matplotlib.pyplot as plt
from generate_spectra import to_spectrum1d, spectra_from_snap, spectra_from_pickle, MUSE_WVL
from scipy.stats import binned_statistic_2d
from spectral_cube import SpectralCube, LazyMask, BooleanArrayMask
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
# import pynbody
from create_noisy_cube import get_header

snap_name = '/home/michele/sim/MySimulations/np_glass/mb.62002_p200_a800_r600/out/snapshot_0048'
size_cuboid=1
sp, (x,y) = spectra_from_snap(snap_name, size_cuboid=size_cuboid, doppler_shift=True)

idx = 1
# plt.step(sp.spectral_axis, sp[idx].flux)


bins=2
a = binned_statistic_2d(x,y, sp.flux.transpose(1,0), statistic='sum', bins=bins, expand_binnumbers=True)

_wcs = WCS(get_header())

mask = LazyMask(lambda x: x>0, data=a.statistic, wcs=_wcs)
cube = SpectralCube(data=a.statistic * u.Unit("erg / (Angstrom cm2 s)"), wcs=_wcs, mask=mask)
cube[:, 0, 0].quicklook()

# Take the first row and column of the bin
row, col = 0, 0
selected_sp = sp[np.where(a.binnumber[row] == col+1)]

# Sum spectra in those bin:
sp_row_col = selected_sp.flux.sum(axis=0)
fig, ax = plt.subplots()
ax.plot(selected_sp.spectral_axis, sp_row_col)

# try to compare but careful to reltol
# np.allclose(sp_row_col.value, cube[:, 0, 0].value)


# Plot all the spectra in row, col
fig, ax = plt.subplots()
for ssp in selected_sp.flux:
   ax.plot(selected_sp.spectral_axis, ssp)

# s = pynbody.load(snap_name)
# guilty = s.s[s.s['iord'] == 1315277]

# fig, ax = plt.subplots()
# ax.plot(cube.spectral_axis, a.statistic[:, 0,0])
plt.show()
