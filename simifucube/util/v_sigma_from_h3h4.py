#!/usr/bin/env python3

import sys
import numpy as np
import os
from astropy.table import Table
import argparse

def vdMarelFranx1993(v, sigma, h3, h4):
    """Convert to true V and Sigma using h3, h4"""
    v_true = v + np.sqrt(3) * sigma * h3
    sigma_true = sigma * (1 + np.sqrt(6) * h4)
    return v_true, sigma_true


def main(cli=None):
    """Should be used with *_ppxf.fits file """
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='ppxf_fits_file', help="Path to *_ppxf.fits file")
    args = parser.parse_args(cli)
    filename = args.ppxf_fits_file
    # dirname = os.path.abspath(sys.argv[1])
    # print(dirname)
    # if not os.path.isdir(dirname):
    #     raise RuntimeError("Input should be the result folder", dirname)
    # filename = os.path.join(dirname, os.path.basename("".join(dirname.split("_")[:-1]) + "_ppxf.fits")
    tbl = Table.read(filename)
    v_true, sigma_true = vdMarelFranx1993(tbl['V'], tbl['SIGMA'], tbl['H3'], tbl['H4'])
    v_true.name = 'V_TRUE'
    sigma_true.name = 'SIGMA_TRUE'
    new_tbl = Table([tbl['BIN_ID'], tbl['V'], v_true, tbl['SIGMA'], sigma_true, tbl['H3'], tbl['H4']])
    # print(new_tbl)
    return new_tbl


if __name__ == '__main__':
    main()
