# This file is part of Simifucube.
#
# Copyright (C) 2013-2019 The Pynbody team (ascl:1305.002, site: https://github.com/pynbody/pynbody)
# Copyright (C) 2019 Michele Mastropietro (michele.mastropietro@gmail.com)
#
# Simifucube is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Simifucube is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Simifucube.  If not, see <https://www.gnu.org/licenses/>.

import pynbody
from pynbody.sph import Kernel
import time
import numpy as np

from simifucube import _render

def render_cube(snap, qty, x2=100, nx=500, y2=None, ny=None, x1=None, y1=None,
                z_plane=0.0, out_units=None, xy_units=None, kernel=Kernel(), z_camera=None,
                smooth='smooth', smooth_in_pixels=False, smooth_range=None, snap_slice=None, num_threads=None
                ):
    """The single-threaded image rendering core function. External calls
    should be made to the render_image function."""

    snap_proxy = {}

    # cache the arrays and take a slice of them if we've been asked to
    for arname in 'x', 'y', 'z', 'pos', smooth, 'rho', 'mass':
        snap_proxy[arname] = snap[arname]
        if snap_slice is not None:
            snap_proxy[arname] = snap_proxy[arname][snap_slice]

    if 'boxsize' in snap.properties:
        boxsize = snap.properties['boxsize'].in_units(snap_proxy['x'].units,**snap.conversion_context())
    else:
        boxsize = None

    in_time = time.time()

    if y2 is None:
        if ny is not None:
            y2 = x2 * float(ny) / nx
        else:
            y2 = x2

    if ny is None:
        ny = nx
    if x1 is None:
        x1 = -x2
    if y1 is None:
        y1 = -y2

    x1, x2, y1, y2, z1 = [float(q) for q in (x1, x2, y1, y2, z_plane)]

    if smooth_range is not None:
        smooth_lo = float(smooth_range[0])
        smooth_hi = float(smooth_range[1])
    else:
        smooth_lo = 0.0
        smooth_hi = 100000.0

    nx = int(nx + .5)
    ny = int(ny + .5)


    n_part = len(snap)

    if xy_units is None:
        xy_units = snap_proxy['x'].units

    x = snap_proxy['x'].in_units(xy_units)
    y = snap_proxy['y'].in_units(xy_units)
    z = snap_proxy['z'].in_units(xy_units)

    sm = snap_proxy[smooth]

    if sm.units != x.units and not smooth_in_pixels:
        sm = sm.in_units(x.units)

    # qty_s = qty
    # qty = snap_proxy[qty]
    mass = snap_proxy['mass']
    rho = snap_proxy['rho']

    if out_units is not None:
        # Calculate the ratio now so we don't waste time calculating
        # the image only to throw a UnitsException later
        conv_ratio = (qty.units * mass.units / (rho.units * sm.units ** kernel.h_power)).ratio(out_units,
                                                                                               **snap.conversion_context())

    if z_camera is None:
        z_camera = 0.0

    if boxsize:
        # work out the tile offsets required to make the image wrap
        num_repeats = int(round(x2/boxsize))+1
        repeat_array = np.linspace(-num_repeats*boxsize,num_repeats*boxsize,num_repeats*2+1)
    else:
        repeat_array = [0.0]

    # print(nx, ny, x, y, z, sm, x1, x2, y1, y2, z_camera, 0.0)
    # print(x1, x2, y1, y2, z_camera)
    # print()
    # print(qty)
    # print()
    # print(smooth_lo, smooth_hi, kernel, repeat_array, repeat_array)
    # for var in (nx, ny, x, y, z, sm, x1, x2, y1, y2, z_camera, 0.0, qty, mass, rho,
    #                               smooth_lo, smooth_hi, kernel, repeat_array, repeat_array):
    #     print(type(var))
    #     if isinstance(var, np.ndarray):
    #         print(var.dtype)

    # print(repeat_array)

    if num_threads is None:
        num_threads = pynbody.config['number_of_threads']
    print('Rendering cube with {} threads'.format(num_threads))
    result = _render.render_cube(nx, ny, x.view(np.ndarray), y.view(np.ndarray), z.view(np.ndarray), sm.view(np.ndarray),
                                 x1, x2, y1, y2, z_camera, 0.0, qty, mass.view(np.ndarray), rho.view(np.ndarray),
                                 smooth_lo, smooth_hi, kernel, num_threads)

    # result = result.view(array.SimArray)

    # The weighting works such that there is a factor of (M_u/rho_u)h_u^3
    # where M-u, rho_u and h_u are mass, density and smoothing units
    # respectively. This is dimensionless, but may not be 1 if the units
    # have been changed since load-time.
    # if out_units is None:
    #     result *= (snap_proxy['mass'].units / (snap_proxy['rho'].units)).ratio(
    #         snap_proxy['x'].units ** 3, **snap_proxy['x'].conversion_context())

    #     # The following will be the units of outputs after the above conversion
    #     # is applied
    #     result.units = snap_proxy[qty_s].units * \
    #         snap_proxy['x'].units ** (3 - kernel.h_power)
    # else:
    #     result *= conv_ratio
    #     result.units = out_units

    # result.sim = snap
    return result