# The module that defines the geometry of the particle, and creates the Langragian markers for the particle. 
# Author: Aditya Tambe

import numpy as np
from numba import njit
import initial

class Particle:
    def __init__(self, x0, y0, nl):

        self.x_centre = x0
        self.y_centre = y0
        self.total_x_distance = 0
        self.phi = 0

        self.Uc_prev, self.Uc_curr, self.Uc_next = 0, 0, 0
        self.Vc_prev, self.Vc_curr, self.Vc_next = 0, 0, 0

        self.omega_z_prev, self.omega_z_curr, self.omega_z_next = 0, 0, 0

        #!! PARTICLE MINOR AXIS AND ASPECT RATIO !!
        self.aspect_ratio = initial.aspect_ratio
        self.minor_axis = initial.minor_axis

        #Array for element lengths in discretization
        self.ds = np.zeros(nl, dtype=np.float64)

        #Arrays for the properties of the particle markers.
        self.x_b = np.zeros(nl, dtype=np.float64)
        self.y_b = np.zeros(nl, dtype=np.float64)
        self.Uw_x = np.zeros(nl, dtype=np.float64)
        self.Uw_y = np.zeros(nl, dtype=np.float64)
        self.ux_b = np.zeros(nl, dtype=np.float64)
        self.uy_b = np.zeros(nl, dtype=np.float64)
        self.rho_b = np.zeros(nl, dtype=np.float64)
        self.Fx_IB = np.zeros(nl, dtype=np.float64)
        self.Fy_IB = np.zeros(nl, dtype=np.float64)

        #These arrays are to mark the node that is closest to the Langragian marker. Each marker
        #gets its own number.
        self.x_left = np.zeros(nl, dtype=np.int32)
        self.x_right = np.zeros(nl, dtype=np.int32)
        self.y_top = np.zeros(nl, dtype=np.int32)
        self.y_bottom = np.zeros(nl, dtype=np.int32)

        #These arrays are to mark the node to where the delta function will reach to for interpolation fluid 
        #velocity. Each marker gets its own number.
        self.x_start = np.zeros(nl, dtype=np.int32)
        self.x_end = np.zeros(nl, dtype=np.int32)
        self.y_start = np.zeros(nl, dtype=np.int32)
        self.y_end = np.zeros(nl, dtype=np.int32)

particle = None

def init_particle(x0, y0, nl):
    global particle
    particle = Particle(x0, y0, nl)

@njit(fastmath=True, nopython=True)
def marker_kernel(
    NL, NX, NY, d_theta, aspect_ratio, minor_axis, delta_range,
    x_centre, y_centre, phi,
    x_b, y_b, x_left, x_right, y_bottom, y_top,
    x_start, x_end, y_start, y_end, ds
):
    for i in range(NL):
        theta = i * d_theta

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x_val = x_centre + aspect_ratio * minor_axis * cos_t * cos_phi - minor_axis * sin_t * sin_phi
        y_val = y_centre + minor_axis * sin_t * cos_phi + aspect_ratio * minor_axis * cos_t * sin_phi

        while x_val >= NX:
            x_val -= NX
        while x_val < 0:
            x_val += NX

        x_b[i] = x_val
        y_b[i] = y_val

        # Base integer node indices (strictly floor)
        x0 = int(np.floor(x_val))
        y0 = int(np.floor(y_val))

        # Assign legacy helper array outputs (if required by signature)
        x_left[i] = x0
        x_right[i] = x0 + 1
        y_bottom[i] = y0
        y_top[i] = y0 + 1

        # Strict 4-point stencil bounds: [x0 - 1, x0, x0 + 1, x0 + 2]
        x_start[i] = x0 - 1
        x_end[i]   = x0 + 2

        ys = y0 - 1
        if ys < 0:
            ys = 0
        y_start[i] = ys

        ye = y0 + 2
        if ye > NY - 1:
            ye = NY - 1
        y_end[i] = ye

        ds_term1 = aspect_ratio * minor_axis * sin_t
        ds_term2 = minor_axis * cos_t
        ds[i] = np.sqrt(ds_term1 * ds_term1 + ds_term2 * ds_term2) * d_theta

def marker():
    p = particle
    marker_kernel(
        initial.NL, initial.NX, initial.NY, initial.d_theta, p.aspect_ratio, p.minor_axis, initial.delta_range,
        p.x_centre, p.y_centre, p.phi,
        p.x_b, p.y_b, p.x_left, p.x_right, p.y_bottom, p.y_top,
        p.x_start, p.x_end, p.y_start, p.y_end, p.ds
    )

@njit(fastmath=True)
def periodic_dx(x_1, x_2):
    dx = abs(x_1 - x_2)
    half_NX = initial.NX/2

    if dx > half_NX:
        dx = initial.NX - dx

    return dx

@njit(fastmath=True)
def wrap(x):
    while (x >= initial.NX):
        x -=  initial.NX

    while (x < 0):
        x += initial.NX

    return x