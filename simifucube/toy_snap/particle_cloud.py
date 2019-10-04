import os
import numpy as np

def create_box_vz_normal(n_star, d, v_mean, sigma):
    vz = np.random.normal(loc=v_mean, scale=sigma, size=n_star)
    x = np.random.uniform(low=-d, high=d, size=n_star)
    y = np.random.uniform(low=-d, high=d, size=n_star)
    z = np.random.uniform(low=-d, high=d, size=n_star)
    pos = np.vstack([x,y,z]).T
    zeros = np.zeros(n_star)
    vel = np.vstack([zeros,zeros,vz]).T

    return pos, vel


def vz_normal(n_star, v_mean, sigma):
    vz = np.random.normal(loc=v_mean, scale=sigma, size=n_star)
    zeros = np.zeros(n_star)
    vel = np.vstack([zeros,zeros,vz]).T
    return vel

def create_sphere_of_positions(n, R):
    phi = np.random.uniform(0, 2 * np.pi, n)
    costheta = np.random.uniform(-1, 1, n)
    u = np.random.uniform(0, 1, n)

    theta = np.arccos(costheta)
    r = R * np.cbrt( u )

    x = r * np.sin( theta) * np.cos( phi )
    y = r * np.sin( theta) * np.sin( phi )
    z = r * np.cos( theta )
    pos = np.vstack([x, y, z]).T
    return pos
