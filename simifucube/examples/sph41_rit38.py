import os

from simifucube.cube_generator import CubeGenerator
from simifucube.generate_spectra import SnapSpectra, muse_rebin

snap_name = os.path.join("/2disk/mmastrop/M1-10_Verbeke2017/M10sim41001/snapshot_0036")

size_cuboid = 1
bins = 80
do_spectral_rebinning=True

out_name = '41_sn36_sph_moria_sum.fits'
pickle_name = 'sn_36_spectra.pkl'

snsp = SnapSpectra(snap_name, size_cuboid=size_cuboid)
#snsp.generate_spectra(save_as=pickle_name)
snsp.generate_spectra_from_pickle(pickle_name)

cg = CubeGenerator(snsp, bins=bins)

cg.sum_on_line_of_sigth()
cube = cg.create_spectral_cube()
muse_rebin(snsp.last_valid_freq, cube).write('pippo_summation.fits', overwrite=True)

im = cg.sph_projection_direct()
cube_sph = cg.create_spectral_cube()

if do_spectral_rebinning:
    muse_cube = muse_rebin(snsp.last_valid_freq, cube_sph)
else:
    muse_cube = cube_sph

muse_cube.write(out_name, overwrite=True)
