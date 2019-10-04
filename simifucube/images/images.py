# Try to get spectrum images from a simulation


import pynbody
from spectra import Spectrum
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from specutils import SpectrumCollection, Spectrum1D
import numpy as np
from parse_filters import Filter
import pickle

plt.style.use('dark_background')


snap_name = '/home/michele/sim/MySimulations/np_glass/mb.62002_p200_a800_r600/out/snapshot_0048'
snap = pynbody.load(snap_name)

# center on the stars
pynbody.analysis.halo.center(snap.s)

size_cuboid = 1

cuboid = pynbody.filt.Cuboid(-size_cuboid)

s = snap[cuboid]

pynbody.analysis.angmom.sideon(s.s)

# Where we look from in kpc
z_dist = 20000

sdss_u = Filter.from_name('u')

print('Redshift:', s.ancestor.header.redshift)

# pos = s.s['pos']
# mass = s.s['mass']
# age = s.s['age']
# met = s.s['metals']
# vlos = s.s['vz']

# img = pynbody.plot.image(s.s, qty='vz')

# TODO Possibility to use SpectrumCollection
spectra = list()
# Here I cannot use directly the iteration over the array s.s because
# by default it iterates over the fields available ['pos', 'vx', ...]
L_sol = 3.839e33  # erg s-1
kpc_in_cm = 3.086e+21  # cm

orig_flux = list()

def run():
    with s.immediate_mode:
        for i in tqdm.tqdm(range(len(s.s))):
            # if i % 100 == 0:
            #   print(i)
            star = s.s[i]
            pos = star['pos'][0].view(np.ndarray)
            dist_sq = (np.linalg.norm(pos - np.array([0, 0, z_dist]))**2)

            # print(star)
            sp1 = Spectrum.from_star(star)
            sp = sp1.doppler_shifted(star['vz'])

            orig_flux.append(sp.flux.copy())
            sp.flux *= star['mass'].in_units('Msol') * L_sol / (4 * np.pi * dist_sq * kpc_in_cm**2)  # erg s-1 cm-2 A-1
            sp.pos = pos
            # sp.intensity = sdss_u.convolve_spectrum(sp) # erg s-1 cm-2
            spectra.append(sp)
    return spectra
spectra = run()


def compute_intensity(spectra, filt):
    intensity = list()
    for sp in spectra:
        intensity.append(sp.convolve(filt))  # erg s-1 cm-2
    return intensity



fig, ax = plt.subplots()

for sp in spectra[:10]:
    sp.plot(ax=ax, linewidth=0.1)



def kpc_to_arcsec(kpc, dist):
    return kpc / dist * (3600 * 180)/np.pi

x = list()
y = list()
intensity = compute_intensity(spectra, sdss_u)

# conversion_factor = kpc_to_arcsec(z_dist)
for sp in spectra:
    x.append(kpc_to_arcsec(sp.pos[0], z_dist))
    y.append(kpc_to_arcsec(sp.pos[1], z_dist))


pickle.dump( (spectra, intensity), open( "spe.pickle", "wb" ) )
fig, ax = plt.subplots()


im = ax.scatter(x, y, s=10, c=intensity, norm=matplotlib.colors.LogNorm(), cmap='gray' )
cbar = fig.colorbar(im)
cbar.ax.set_ylabel('Flux [erg s$^{-1}$ cm$^{-2}$]')
ax.set_title("{}-band flux at {} Mpc".format(sdss_u.name, z_dist/1000))
ax.set_xlabel('x [arcsec]')
ax.set_ylabel('y [arcsec]')
