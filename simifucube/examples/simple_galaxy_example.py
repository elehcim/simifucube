import os

import pynbody

from simifucube.cube_generator import CubeGenerator
from simifucube.generate_spectra import SnapSpectra, muse_rebin
from simifucube.write_cube import write_cube

MORIA_PATH='/mnt/data/MoRIA/M1-10_Verbeke2017/M10sim41001'
SIM_PATH = '/home/michele/sim/MySimulations/ng'
sim = 62002
peri = 200
isnap = 227

_do_preprocessing=True
num_threads=1
moria=False
rit38=False
toy = True


if moria:
    snap_name = "/mnt/data/MoRIA/M1-10_Verbeke2017/M01sim14001/snapshot_0000"
    # snap_name = os.path.join(MORIA_PATH, 'snapshot_0036')
    out_name = 'M01sn0'
elif rit38:
    snap_name = os.path.join("/2disk/mmastrop/M1-10_Verbeke2017/M10sim41001/snapshot_0036")
elif toy:
    snap_name = 'toy_snap_d4.0'
    pynbody.config['sph']['smooth-particles'] = 2
    pynbody.config['sph']['tree-leafsize'] = 1
    out_name='toycubepix20METm1.5AGE1-4-Vm20p500d4.0'
    _do_preprocessing=False
else:
    snap_name = os.path.join(SIM_PATH, "mb.{}_p{}_a800_r600/out/snapshot_{:04d}".format(sim, peri, isnap))
    out_name = '{}p{}_sn{}'.format(62002, peri, isnap)

size_cuboid = 3
bins = 20
do_spectral_rebinning=True

pickle_name = 'test_pickle.pkl'

snsp = SnapSpectra(snap_name, size_cuboid=size_cuboid, do_preprocessing=_do_preprocessing)
snsp.generate_spectra(use_template_star=True)
# snsp.generate_spectra_from_pickle(pickle_name)

cg = CubeGenerator(snsp, bins=bins)
# cg.sum_on_line_of_sigth()
# cube = cg.create_spectral_cube()
# cube.write(out_name+'pix{}_sum.fits'.format(bins), overwrite=True)

im = cg.sph_projection_direct(num_threads=num_threads)

cube_sph = cg.create_spectral_cube()

if do_spectral_rebinning:
    muse_cube = muse_rebin(snsp.last_valid_freq, cube_sph)
else:
    muse_cube = cube_sph


# muse_cube.write(out_name+'pix{}.fits'.format(bins), overwrite=True)

write_cube(muse_cube, variance_cube=None, filename=out_name+'pix{}.fits'.format(bins), overwrite=True)