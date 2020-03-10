# This file is part of Simifucube.
#
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

import os
import subprocess
from astropy.io import fits

def contract_name(stem, sim, isnap, peri, r, ext, bins=None, fix=False, doppler_shift=True, **kwargs):
    sim_str = str(sim)[:2]
    f = "{}_{}p{:g}_{:04d}_r{}".format(stem, sim_str, peri/100, isnap, r)
    if bins is not None:
        f += "_b{}".format(bins)
    if not doppler_shift:
        f += "_nds"
    if fix:
        f += "_fix"
    for k, v in kwargs.items():
        f += "_{}{}".format(k, v)
    f += ".{}".format(ext)
    return f


def cd_keyword(header):
    for i in range(1, 4):
        header['CD{0}_{0}'.format(i)] = header['CDELT{}'.format(i)]
    return header


def get_git_version():
    label = ''
    cwd = os.path.dirname(__file__)
    try:
        label = subprocess.check_output(["git", "describe", "--tags", "--dirty"],
                                        stderr=subprocess.STDOUT,
                                        cwd=cwd).strip().decode()
    except subprocess.CalledProcessError:
        print(label)
        label = 'NO_VERSION'
    return label

def sanitize_cards(d, max_len=50):
    meta = d.copy()
    new_keys = dict()
    doomed_keys = list()
    for k, v in meta.items():
        if len(v) > max_len:
            doomed_keys.append(k)
            i = 1
            while len(v) > max_len:
                new_keys[k + f"_{i}"] = v[: max_len]
                v = v[max_len:]
            # last piece:
            new_keys[k + f"_{i}"] = v
    # print(doomed_keys)
    for _k in doomed_keys:
        del meta[_k]
    meta.update(new_keys)
    return meta

def write_cube(cube, variance_cube, filename, overwrite=False, meta=None):
    print('Writing cube {}'.format(filename))
    hdulist = fits.HDUList([fits.PrimaryHDU()])
    hdulist[0].header['SWMASTRO'] = get_git_version()
    if meta is not None and isinstance(meta, dict):
        mymeta = sanitize_cards(meta)
        for k, v in mymeta.items():
            hdulist[0].header[k] = v

    hdulist.append(cube.hdu)
    hdulist[1].name = 'DATA'
    hdulist[1].header = cd_keyword(hdulist[1].header)
    if variance_cube is not None:
        hdulist.append(variance_cube.hdu)
        hdulist[2].name = 'STAT'
        hdulist[2].header = cd_keyword(hdulist[2].header)

    print(hdulist.info())
    hdulist.writeto(filename, overwrite=overwrite)
