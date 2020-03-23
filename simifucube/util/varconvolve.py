# Copyright 2016-2020 Janez Kos, Michele Mastropietro
# Adapted from https://github.com/sheliak/varconvolve
# Licensed under GPLv3

import numpy as np
from scipy.interpolate import interp1d
import types
from numba import jit
# To remove dependency on scipy
# def interp1d(x, arr):
#     return np.interp(x, np.arange(stop=arr.size), arr)

@jit
def _gaussian_kernel(s):
    """
    Constructs a normalized discrete 1D gaussian kernel
    """
    size_grid = int(s*4)
    # x= np.mgrid[-size_grid:size_grid+1]
    # print(x)

    x = np.arange(-size_grid, size_grid+1)
    # print(x)
    g = np.exp(-(x**2/float(s**2)/2.))
    return g / np.sum(g)


def varconvolve(x, y, kernel, var, oversample=1, mode='same'):
    """
    x: an array with x coordinates of the points
    y: an array with y coordinates of the points
    kernel: name of the function that describes the kernel. Must have one argument, the width of the kernel in x units
    var: a function that returns the kernel width in one point.
    """
    # print('.', end='')
    # if isinstance(var, (types.FunctionType)):
    #     pass
    # elif isinstance(var, (list, tuple, np.ndarray)):
    #     var = interp1d(x, var)
    # else:
    #     raise RuntimeError('var must be a function, a list, a tuple, or an ndarray')

    # check if sampling is uniform:
    if abs(np.max(np.diff(x)) - np.min(np.diff(x))) < 0.000001 * np.diff(x)[0]:
        sampl = np.diff(x)[0]
    else:
        raise RuntimeError('Sampling must be uniform.')

    # build the new sampling array. Start at first element in x, end after we reach the last element in x:
    x_new = [x[0]]
    n = 0

    m = np.max(var(x))

    # In practice x_new[n+1] - x_new[x] = var(x_new[n]) * dx_old * factor
    while x_new[n] + var(x_new[n]) / m / oversample * sampl <= x[-1]:
        x_new.append(x_new[n] + var(x_new[n]) / m / oversample * sampl)
        n += 1
    x_new.append( x_new[n] + var(x_new[n]) / m / oversample * sampl)

    if len(kernel(m * oversample / sampl)) > len(x_new):
        raise RuntimeError('Kernel is larger than the data range.')

    y_new = np.interp(x_new,x,y)

    y_con = np.convolve(y_new, kernel(m * oversample / sampl), mode=mode)

    y_out = np.interp(x,x_new,y_con)

    return y_out





if __name__ == '__main__':
    import numpy as np
    import scipy
    from matplotlib.pyplot import *

    def kernel(s):
        """
        Constructs a normalized discrete 1D gaussian kernel
        """
        size_grid = int(s*4)
        x= scipy.mgrid[-size_grid:size_grid+1]
        g = scipy.exp(-(x**2/float(s**2)/2.))
        return g / np.sum(g)

    def var(x):
        """
        Creates a polynomial that describes the kernel width
        """
        p=[0.2,0.02]
        return np.polyval(p,x)

    #x sampling:
    x=np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])
    #a delta function:
    y=np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
    #a step function that describes the kernel width:
    v=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3])

    plot(x,var(x),'k-')
    y_c=varconvolve(x,y,kernel,var)
    y_cc=varconvolve(x,y,kernel,v)
    plot(x,y_c,'r-')
    plot(x,y_cc,'b-')

    show()