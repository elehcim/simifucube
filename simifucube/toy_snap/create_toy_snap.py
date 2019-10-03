import pynbody
import numpy as np
from pynbody.array import SimArray

f = pynbody.new(star=2)

mass_prefactor = 1e4

f['pos'] = SimArray(np.array([[-2, 0, 0], [2,0,0]]), units='kpc')
f['vel'] = SimArray(np.array([[0, 0, -20],[0,0,500]]), units='km s**-1')
f['mass'] = SimArray([1e-10,1e-10], units='1e10 Msol') * mass_prefactor
f.properties['boxsize']=1000 * pynbody.units.kpc
print(f.properties)

pynbody.config['sph']['smooth-particles'] = 1
pynbody.config['sph']['tree-leafsize'] = 1
f['smooth'] = SimArray(np.array([0.2, 0.2]), units='kpc')

f.write(filename='toy_snap_d4.0_M1e4', fmt=pynbody.snapshot.gadget.GadgetSnap)
