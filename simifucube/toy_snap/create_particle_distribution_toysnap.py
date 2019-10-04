import os
import pynbody
import numpy as np
from pynbody.array import SimArray
from particle_cloud import  create_sphere_of_positions, create_box_vz_normal, vz_normal


config = dict(
n_star=1000,  # number of stars per bunch
r_sphere=2,   # radius of the spherical cloud

particle_mass = 1e4,  # Msun
boxsize = 1000,
overwrite=True,
stem='toy_snap_cloud',
v1=20,
v2=-500,
sigma=10,
dist=2,
)


STD_PROPERTIES = dict(a=1.0,
                      z=0,
                      boxsize=1000*pynbody.units.kpc,
                      h=0.7,
                      omegaL0=0.72,
                      omegaM0=1.0,
                      time=0 * pynbody.units.Unit("s kpc km**-1"))

def get_char_sign(x):
    return 'm' if x < 0 else 'p'

# Create clouds of stars uniformly distributed in a sphere with a gaussian distribution of velocities around a given mean.


def get_all_keys(snap):
    """return all the (non derived) keys for all the families"""
    ak = set()
    for fam in snap.families():
        ak = ak.union(snap[fam].loadable_keys()).union(snap[fam].keys()).union(snap[fam].family_keys())
    ak = [k for k in ak if not k in ["x", "y", "z", "vx", "vy", "vz"]]
    ak.sort()
    return ak

def create_snap(n_star, r_sphere, v, sigma, particle_mass, **kwargs):
    f = pynbody.new(star=n_star)

    pos = create_sphere_of_positions(n_star, r_sphere)
    vel = vz_normal(n_star, v, sigma)
    f['pos'] = SimArray(pos, units='kpc')
    f['vel'] = SimArray(vel, units='km s**-1')
    f['mass'] = SimArray([1e-10]*n_star, units='1e10 Msol') * particle_mass
    return f

def join_snaps(s1, s2):
    s = pynbody.snapshot.new(dm=len(s1.dm)+len(s2.dm),
                             gas=len(s1.gas)+len(s2.gas),
                             star=len(s1.s)+len(s2.s),
                             order='gas,dm,star')

    def stack_3vec(s1, s2, key):
        if key in ['pos', 'vel']:
            stacked = pynbody.array.SimArray(np.vstack([s1[key], s2[key]]), s1[key].units)
        else:
            stacked = pynbody.array.SimArray(np.hstack([s1[key], s2[key]]), s1[key].units)
        return stacked

    for k in get_all_keys(s1.gas):
        print("Stacking ", k)
        stacked = stack_3vec(s1.g, s2.g, k)
        s.g[k] = stacked

    for k in get_all_keys(s1.dm):
        print("Stacking ", k)
        stacked = stack_3vec(s1.dm, s2.dm, k)
        s.dm[k] = stacked

    for k in get_all_keys(s1.s):
        print("Stacking ", k)
        stacked = stack_3vec(s1.s, s2.s, k)
        s.s[k] = stacked

    # s.properties = s1.properties.copy()
    s.properties = STD_PROPERTIES
    # s.properties['z'] = s1.properties['z']
    # s.properties['a'] = s1.properties['a']
    # s.properties['omegaM0'] = s1.properties['omegaM0']
    # s.properties['omegaM0'] = s1.properties['omegaM0']

    return s

def main(config):
    n_star = config['n_star']
    stem = config['stem']

    v1_sign = get_char_sign(config['v1'])
    v2_sign = get_char_sign(config['v2'])

    outname = f"{stem}_n{n_star}V{v1_sign}{config['v1']}{v2_sign}{config['v2']}_s{config['sigma']}.snap"
    print(f"Writing to {outname}")
    if os.path.isfile(outname):
        if config['overwrite']:
            print(f"Removing {outname} to create a new one...")
            os.remove(outname)
        else:
            raise RuntimeError("File {outname} already exists. Use 'overwrite' or use a different name")


    f1 = create_snap(v=config['v1'], **config)
    f2 = create_snap(v=config['v2'], **config)

    # offset along x
    f1['pos'][:,0] -= config['dist']/2
    f2['pos'][:,0] += config['dist']/2

    f = join_snaps(f1, f2)

    f.properties['boxsize'] = config['boxsize'] * pynbody.units.kpc
    print(f.properties)
    # print(f['smooth'])
    # pynbody.config['sph']['smooth-particles'] = 1
    # pynbody.config['sph']['tree-leafsize'] = 1
    print(len(f.s))
    # f['smooth'] = SimArray(np.array([0.2]*len(f)), units='kpc')

    f.write(filename=outname, fmt=pynbody.snapshot.gadget.GadgetSnap)
    return f


if __name__ == '__main__':
    f = main(config)