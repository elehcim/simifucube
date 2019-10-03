import numpy as np

cimport numpy as np
cimport cython
from cython.parallel import prange, threadid

from libc.stdio cimport printf

cimport libc.math as cmath
from libc.math cimport atan, pow
from libc.stdlib cimport malloc, free

# The following slightly odd repetitiveness is to force Cython to generate
# code for different permutations of the possible integer inputs.
#
# Using just one fused type requires the types to be consistent across all
# arguments.

ctypedef fused fused_input_type_1:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_2:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_3:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_4:
    np.float32_t
    np.float64_t

ctypedef fused fused_input_type_5:
    np.float32_t
    np.float64_t


ctypedef np.float32_t image_output_type
np_image_output_type = np.float32

ctypedef np.float64_t fixed_input_type


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef image_output_type get_kernel(fixed_input_type d2, fixed_input_type kernel_max_2,
                                  image_output_type h_to_the_kdim, int num_samples,
                                  image_output_type* kvals) nogil :
    cdef unsigned int index = <unsigned int>(num_samples*(d2/kernel_max_2))
    if index<num_samples :
        return kvals[index]/h_to_the_kdim
    else :
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef image_output_type get_kernel_xyz(fixed_input_type x, fixed_input_type y, fixed_input_type z, fixed_input_type kernel_max_2,
                                       image_output_type h_to_the_kdim, int num_samples,
                                 image_output_type* kvals) nogil :
     return get_kernel(x*x+y*y+z*z,kernel_max_2,h_to_the_kdim,num_samples,kvals)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_cube(int nx, int ny,  # number of pixel
                 np.ndarray[fused_input_type_1,ndim=1] x,   # array of particle positions x
                 np.ndarray[fused_input_type_1,ndim=1] y,   # array of particle positions y
                 np.ndarray[fused_input_type_1,ndim=1] z,   # array of particle positions z
                 np.ndarray[fused_input_type_2,ndim=1] sm,  # particle smoothing length
                 fixed_input_type x1, fixed_input_type x2,  # The x-coordinate of the left (x1) and right (x2) edge of the image
                 fixed_input_type y1, fixed_input_type y2,  # The y-coordinate of the bottom (y1 and top (y2) edge of the image
                 fixed_input_type z_camera, fixed_input_type z0,
                 np.ndarray[fused_input_type_3,ndim=2] qty, # array of quantity to make image of
                 np.ndarray[fused_input_type_4,ndim=1] mass,# array of particle masses
                 np.ndarray[fused_input_type_5,ndim=1] rho, # array of particle densities
                 fixed_input_type smooth_lo, fixed_input_type smooth_hi,
                 kernel, int num_threads) :

    cdef fixed_input_type pixel_dx = (x2-x1)/nx
    cdef fixed_input_type pixel_dy = (y2-y1)/ny
    cdef fixed_input_type x_start = x1+pixel_dx/2
    cdef fixed_input_type y_start = y1+pixel_dy/2
    cdef int n_part = len(x)
    cdef int n_channel = qty.shape[1]
    cdef int nn=0, i=0, j=0
    cdef fixed_input_type x_i, y_i, z_i, sm_i#, qty_i
    cdef fixed_input_type x_pixel, y_pixel, z_pixel
    cdef int x_pos, y_pos
    cdef int x_pix_start, x_pix_stop, y_pix_start, y_pix_stop

    # following are only used for "perspective" rendering
    # cdef float per_z_dx = (x2-x1)/(2*z_camera)
    # cdef float per_z_dy = (y2-y1)/(2*z_camera)
    # cdef float mid_x = (x2+x1)/2
    # cdef float mid_y = (y2+y1)/2
    # cdef float dz_i

    cdef int kernel_dim = kernel.h_power
    cdef fixed_input_type max_d_over_h = kernel.max_d


    cdef np.ndarray[image_output_type,ndim=1] samples = kernel.get_samples(dtype=np_image_output_type)
    cdef int num_samples = len(samples)
    cdef image_output_type* samples_c = <image_output_type*>samples.data
    cdef image_output_type sm_to_kdim   # minimize casting when same type as output

    cdef fixed_input_type kernel_max_2 # minimize casting when same type as input

    # cdef np.ndarray[image_output_type,ndim=3] result = np.zeros((ny, nx, n_channel), dtype=np_image_output_type)

    z_pixel = z0
    # cdef int total_ptcls = 0

    cdef int use_z = 1 if kernel_dim>=3 else 0

    assert kernel_dim==2 or kernel_dim==3, "Only kernels of dimension 2 or 3 currently supported"
    assert len(x) == len(y) == len(z) == len(sm) == len(qty) == len(mass) == len(rho), "Inconsistent array lengths passed to render_image_core"
    # cdef fixed_input_type *qty_i = <fixed_input_type *> malloc(n_channel * sizeof(fixed_input_type))
    # cdef np.ndarray[image_output_type,ndim=2] qty_i = np.zeros((n_part, n_channel), dtype=np_image_output_type)

    printf("kernel_dim=%d\n",  kernel_dim)
    printf("n_part=%d\n",  n_part)
    printf("n_channel=%d\n", n_channel)
    printf("nx=%d\n", nx)
    printf("ny=%d\n", ny)
    printf("x1=%f\n", x1)
    printf("x2=%f\n", x2)
    printf("y1=%f\n", y1)
    printf("y2=%f\n", y2)

    printf("use_z=%d\n", use_z)
    printf("z_camera=%f\n", z_camera)

    # printf("wrap_offsets_x=%f\n", wrap_offsets_x)
    # printf("wrap_offsets_y=%f\n", wrap_offsets_y)

    cdef int tid
    # Inspired from here: https://stackoverflow.com/a/42283906/
    cdef np.ndarray[image_output_type,ndim=4] result_local = np.zeros((num_threads, ny, nx, n_channel), dtype=np_image_output_type)

    cdef fixed_input_type *qty_i = <fixed_input_type *> malloc(num_threads * n_channel * sizeof(fixed_input_type))

    for i in prange(n_part, nogil=True, num_threads=num_threads):
        tid = threadid()

        if i % 100 == 0:
            printf("Particle %d/%d (%.2f%%)\r", i, n_part, (<double>i)/n_part*100)

        # load particle details
        x_i = x[i]
        y_i = y[i]
        z_i = z[i]
        sm_i = sm[i]
        for j in range(n_channel):
            qty_i[tid*n_channel + j] = qty[i, j]*mass[i]/rho[i]

        # if z_camera!=0.0 :
        #     # perspective image -
        #     # update image bounds for the current z
        #     if (z_i>z_camera and z_camera>0) or (z_i<z_camera and z_camera<0) :
        #         # behind camera
        #         continue
        #     dz_i = z_camera-z_i
        #     x1 = mid_x - per_z_dx*dz_i
        #     x2 = mid_x + per_z_dx*dz_i
        #     y1 = mid_y - per_z_dy*dz_i
        #     y2 = mid_y + per_z_dy*dz_i
        #     pixel_dx = (x2-x1)/nx
        #     pixel_dy = (y2-y1)/ny
        #     x_start = x1+pixel_dx/2
        #     y_start = y1+pixel_dy/2


        # check particle smoothing is within specified range
        if sm_i<pixel_dx*smooth_lo or sm_i>pixel_dx*smooth_hi :
            printf("Particle %d smoothing not within imposed range of (%.2f, %.2f)\n", i, smooth_lo, smooth_hi)
            continue

        # total_ptcls+=1

        # check particle is within bounds
        # printf("=====Particle %d\n",i)
        # printf("x1=%f\n", x1)
        # printf("x2=%f\n", x2)
        # printf("y1=%f\n", y1)
        # printf("y2=%f\n", y2)
        # printf("x_i=%f\n",x_i)
        # printf("y_i=%f\n",y_i)
        # printf("sm_i=%f\n",sm_i)

        # printf("Test  1 %d\n",use_z*cmath.fabs(z_i-z0)<max_d_over_h*sm_i)
        # printf("x1-2*sm_i %f\n",x1-2*sm_i)
        # printf("Test x1 %d\n",x_i>(x1-2*sm_i))
        # printf("Test x2 %d\n",x_i<x2+2*sm_i)
        # printf("Test y1 %d\n",y_i>y1-2*sm_i)
        # printf("Test y2 %d\n",y_i<y2+2*sm_i)

        if not ((use_z*cmath.fabs(z_i-z0)<max_d_over_h*sm_i)
                and x_i>x1-2*sm_i and x_i<x2+2*sm_i and y_i>y1-2*sm_i and y_i<y2+2*sm_i) :
            printf("Particle %d not within bounds\n",i)
            continue

        # pre-cache sm^kdim and (sm*max_d_over_h)**2; tests showed massive speedups when doing this
        if kernel_dim==2 :
            sm_to_kdim = sm_i*sm_i
        else :
            sm_to_kdim = sm_i*sm_i*sm_i
            # only 2, 3 supported

        kernel_max_2 = (sm_i*sm_i)*(max_d_over_h*max_d_over_h)


        # decide whether this is a single pixel or a multi-pixel particle
        if (max_d_over_h*sm_i/pixel_dx<1 and max_d_over_h*sm_i/pixel_dy<1) :
            # printf("Single pixel Particle %d\n",i)
            # single pixel, get pixel location
            x_pos = int((x_i-x1)/pixel_dx)
            y_pos = int((y_i-y1)/pixel_dy)

            # work out pixel centre
            x_pixel = (pixel_dx*<fixed_input_type>(x_pos)+x_start)
            y_pixel = (pixel_dy*<fixed_input_type>(y_pos)+y_start)

            # final bounds check
            if x_pos>=0 and x_pos<nx and y_pos>=0 and y_pos<ny :
                for j in range(n_channel):
                    result_local[tid,y_pos,x_pos,j]+=qty_i[tid*n_channel + j]*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)
        else :
            # printf("Multi-pixel Particle %d \n",i)
            # multi-pixel
            x_pix_start = int((x_i-max_d_over_h*sm_i-x1)/pixel_dx)
            x_pix_stop =  int((x_i+max_d_over_h*sm_i-x1)/pixel_dx)
            y_pix_start = int((y_i-max_d_over_h*sm_i-y1)/pixel_dy)
            y_pix_stop =  int((y_i+max_d_over_h*sm_i-y1)/pixel_dy)
            if x_pix_start<0 : x_pix_start = 0
            if x_pix_stop>nx : x_pix_stop = nx
            if y_pix_start<0 : y_pix_start = 0
            if y_pix_stop>ny : y_pix_stop = ny
            for y_pos in range(y_pix_start, y_pix_stop) :
                y_pixel = pixel_dy*<fixed_input_type>(y_pos)+y_start
                for x_pos in range(x_pix_start, x_pix_stop) :
                    x_pixel = pixel_dx*<fixed_input_type>(x_pos)+x_start

                    # could accessing the buffer manually be
                    # faster? It seems to be FAR faster (x10!) but
                    # only when using stack-allocated memory for
                    # c_result, and when writing to memory, not
                    # also reading (i.e. = instead of +=).  The
                    # instruction that, according to Instruments,
                    # holds everything up and disappears is
                    # cvtss2sd, but it's not clear to me why this
                    # disappears from the compiled version in the
                    # instance described above.  Anyway for now, there
                    # is no advantage to the manual approach -

                    #c_result[x_pos+nx*y_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)
                    for j in range(n_channel):
                        result_local[tid,y_pos,x_pos,j]+=qty_i[tid*n_channel + j]*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)

    free(qty_i)

    printf("\nSPH rendering done\n")
    return result_local.sum(axis=0)




# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# def to_3d_grid(int nx, int ny, int nz,
#                  np.ndarray[fused_input_type_1,ndim=1] x,
#                  np.ndarray[fused_input_type_1,ndim=1] y,
#                  np.ndarray[fused_input_type_1,ndim=1] z,
#                  np.ndarray[fused_input_type_2,ndim=1] sm,
#                  fixed_input_type x1,fixed_input_type x2,fixed_input_type y1,
#                  fixed_input_type y2,fixed_input_type z1, fixed_input_type z2,
#                  np.ndarray[fused_input_type_3,ndim=1] qty,
#                  np.ndarray[fused_input_type_4,ndim=1] mass,
#                  np.ndarray[fused_input_type_5,ndim=1] rho,
#                  fixed_input_type smooth_lo, fixed_input_type smooth_hi,
#                  kernel) :



#     cdef fixed_input_type pixel_dx = (x2-x1)/nx
#     cdef fixed_input_type pixel_dy = (y2-y1)/ny
#     cdef fixed_input_type pixel_dz = (z2-z1)/ny
#     cdef fixed_input_type x_start = x1+pixel_dx/2
#     cdef fixed_input_type y_start = y1+pixel_dy/2
#     cdef fixed_input_type z_start = z1+pixel_dz/2
#     cdef int n_part = len(x)
#     cdef int nn=0, i=0
#     cdef fixed_input_type x_i, y_i, z_i, sm_i, qty_i
#     cdef fixed_input_type x_pixel, y_pixel, z_pixel
#     cdef int x_pos, y_pos, z_pos
#     cdef int x_pix_start, x_pix_stop, y_pix_start, y_pix_stop, z_pix_start, z_pix_stop

#     cdef int kernel_dim = kernel.h_power
#     cdef fixed_input_type max_d_over_h = kernel.max_d

#     cdef np.ndarray[image_output_type,ndim=1] samples = kernel.get_samples(dtype=np_image_output_type)
#     cdef int num_samples = len(samples)
#     cdef image_output_type* samples_c = <image_output_type*>samples.data
#     cdef image_output_type sm_to_kdim   # minimize casting when same type as output

#     cdef fixed_input_type kernel_max_2 # minimize casting when same type as input

#     cdef np.ndarray[image_output_type,ndim=3] result = np.zeros((nx,ny,nz),dtype=np_image_output_type)

#     cdef int total_ptcls = 0

#     cdef int use_z = 1 if kernel_dim>=3 else 0

#     if kernel_dim<3:
#         raise ValueError, \
#           "Cannot render to 3D grid without 3-dimensional kernel or greater"

#     assert len(x) == len(y) == len(z) == len(sm) == \
#             len(qty) == len(mass) == len(rho), \
#             "Inconsistent array lengths passed to render_image_core"

#     with nogil:
#         for i in range(n_part) :
#             # load particle details
#             x_i = x[i]; y_i=y[i]; z_i=z[i]; sm_i = sm[i];
#             qty_i = qty[i]*mass[i]/rho[i]

#             # check particle smoothing is within specified range
#             if sm_i<pixel_dx*smooth_lo or sm_i>pixel_dx*smooth_hi : continue

#             total_ptcls+=1

#             # check particle is within bounds
#             if not (z_i>z1-2*sm_i and z_i<z2+2*sm_i \
#                     and x_i>x1-2*sm_i and x_i<x2+2*sm_i \
#                     and y_i>y1-2*sm_i and y_i<y2+2*sm_i) :
#                 continue

#             # pre-cache sm^kdim and (sm*max_d_over_h)**2; tests showed massive speedups when doing this
#             if kernel_dim==2 :
#                 sm_to_kdim = sm_i*sm_i
#             else :
#                 sm_to_kdim = sm_i*sm_i*sm_i
#                 # only 2, 3 supported

#             kernel_max_2 = (sm_i*sm_i)*(max_d_over_h*max_d_over_h)

#             # decide whether this is a single pixel or a multi-pixel particle
#             if (max_d_over_h*sm_i/pixel_dx<1 and max_d_over_h*sm_i/pixel_dy<1) :
#                 # single pixel, get pixel location
#                 x_pos = int((x_i-x1)/pixel_dx)
#                 y_pos = int((y_i-y1)/pixel_dy)
#                 z_pos = int((z_i-z1)/pixel_dz)

#                 # work out pixel centre
#                 x_pixel = (pixel_dx*<fixed_input_type>(x_pos)+x_start)
#                 y_pixel = (pixel_dy*<fixed_input_type>(y_pos)+y_start)
#                 z_pixel = (pixel_dz*<fixed_input_type>(z_pos)+z_start)

#                 # final bounds check
#                 if x_pos>=0 and x_pos<nx and y_pos>=0 and y_pos<ny \
#                    and z_pos>=0 and z_pos<nz :
#                     result[x_pos,y_pos,z_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel)*use_z, kernel_max_2 ,sm_to_kdim,num_samples,samples_c)
#             else :
#                 # multi-pixel
#                 x_pix_start = int((x_i-max_d_over_h*sm_i-x1)/pixel_dx)
#                 x_pix_stop =  int((x_i+max_d_over_h*sm_i-x1)/pixel_dx)
#                 y_pix_start = int((y_i-max_d_over_h*sm_i-y1)/pixel_dy)
#                 y_pix_stop =  int((y_i+max_d_over_h*sm_i-y1)/pixel_dy)
#                 z_pix_start = int((z_i-max_d_over_h*sm_i-z1)/pixel_dz)
#                 z_pix_stop =  int((z_i+max_d_over_h*sm_i-z1)/pixel_dz)
#                 if x_pix_start<0 : x_pix_start = 0
#                 if x_pix_stop>nx : x_pix_stop = nx
#                 if y_pix_start<0 : y_pix_start = 0
#                 if y_pix_stop>ny : y_pix_stop = ny
#                 if z_pix_start<0 : z_pix_start = 0
#                 if z_pix_stop>nz : z_pix_stop = nz
#                 for x_pos in range(x_pix_start, x_pix_stop) :
#                     x_pixel = pixel_dx*<fixed_input_type>(x_pos)+x_start
#                     for y_pos in range(y_pix_start, y_pix_stop) :
#                         y_pixel = pixel_dy*<fixed_input_type>(y_pos)+y_start

#                         for z_pos in range(z_pix_start,z_pix_stop) :
#                             z_pixel = pixel_dz*<fixed_input_type>(z_pos)+z_start
#                             result[x_pos,y_pos,z_pos]+=qty_i*get_kernel_xyz(x_i-x_pixel, y_i-y_pixel, (z_i-z_pixel), kernel_max_2 ,sm_to_kdim,num_samples,samples_c)

#     return result
