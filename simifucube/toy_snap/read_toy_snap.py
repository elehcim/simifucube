import pynbody
import numpy as np
from pynbody.array import SimArray

f = pynbody.load('toy_snap')
pynbody.config['sph']['smooth-particles'] = 2
pynbody.config['sph']['tree-leafsize'] = 1

f['smooth']