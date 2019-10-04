import pynbody
import numpy as np
from pynbody.array import SimArray

f = pynbody.new(star=2)

mass_prefactor = 1e4

f['pos'] = SimArray(np.array([[-1, 0, 0], [2,0,0]]), units='kpc', dtype=np.float32)
f['vel'] = SimArray(np.array([[0, 0, -20],[0,0,500]]), units='km s**-1')
f['mass'] = SimArray([1e-10,1e-10], units='1e10 Msol') * mass_prefactor
f.properties['boxsize']=1000 * pynbody.units.kpc
print(f.properties)

pynbody.config['sph']['smooth-particles'] = 1
pynbody.config['sph']['tree-leafsize'] = 1
f['smooth'] = SimArray(np.array([1.0, 1.0]), units='kpc')

filename = 'toy_snap_d3_M1e4_sm1'
f.write(filename=filename, fmt=pynbody.snapshot.gadget.GadgetSnap)
print(f"Created {filename}")