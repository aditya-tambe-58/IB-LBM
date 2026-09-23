# The initial conditions file for the IB-LBM simulation. 
# Author: Aditya Tambe

import numpy as np

#Define the Reynolds number and the initial position of the particle.
Re = 40
X0 = 100
Y0 = 128

#Number of Lnagragian markers.
NL = 235

#What's this?
Ux_b = 0.0
Uy_b = 0.0

#Define simulation and particle constants.
minor_axis = 25
aspect_ratio = 1
NX = 801
NY = 202
Q = 9

# Semi-axes
a = aspect_ratio * minor_axis
b = minor_axis

t_max = 1000
rho_0 = 1
H = NY - 2
tau = 0.75
omega = 1/tau
visc = (tau - 0.5)/3
u_mean = (Re * visc) / H
u_max = 1.5 * u_mean
Ma = u_max/0.57735

delta_range = 2

fx_ext = (12 * (visc**2) * Re) / (H**3)
fy_ext = 0

ellipse_area = np.pi * a * b

# Masses and Moments of Inertia
rho_s = 1.0
rho_f = rho_0

M_s = rho_s * ellipse_area
M_f = rho_f * ellipse_area

I_s = 0.25 * M_s * (a**2 + b**2)
I_f = 0.25 * M_f * (a**2 + b**2)

w0 = 4/9
w1 = 1/9
w2 = 1/36

cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
sym = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=np.int32)
t = np.array([w0, w1, w1, w1, w1, w2, w2, w2, w2], dtype=np.float64)

d_theta = (2 * np.pi) / NL


