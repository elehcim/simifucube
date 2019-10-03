import pynbody
import numpy as np
from pynbody.array import SimArray

f = pynbody.new(star=8)

mass_prefactor = 1e4

f['pos'] = SimArray(np.array([
    [-2,  2, 2], [-2,  2, -2],
    [-2, -2, 2], [-2, -2, -2],
    [ 2, -2, 2], [ 2, -2, -2],
    [ 2,  2, 2], [ 2,  2, -2]]
    )*0.8, units='kpc')

f['vel'] = SimArray(np.array([[0, 0, -60],[0,0,80]]*4), units='km s**-1')

f['mass'] = SimArray([1e-10]*8, units='1e10 Msol') * mass_prefactor
# f.properties['time'] = 14 * pynbody.units.Unit('s kpc km**-1')

# the 1 minus part is because in the output time is set to 1 in internal units , so that the age is 1 minus the tform.
f['tform'] = 1 * pynbody.units.Unit('s kpc km**-1') - SimArray([1,1]*2+[1,4]*2, units='Gyr')
print(f['tform'])
f['feh'] = SimArray([-1.22,-1.22]+[0.29, -1.22]*2 + [-1.22, -1.22])

f.properties['boxsize'] = 1000 * pynbody.units.kpc
print(f.properties)

pynbody.config['sph']['smooth-particles'] = 1
pynbody.config['sph']['tree-leafsize'] = 1
f['smooth'] = SimArray(np.array([0.2]*8), units='kpc')

f.write(filename='toy_snap_multi_Vm60p80', fmt=pynbody.snapshot.gadget.GadgetSnap)
