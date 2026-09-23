# The module that couples the particle to the fluid using the Immersed Boundary Method. 
# Author: Aditya Tambe

import numpy as np
from numba import njit, prange
import initial
import geometry
import lbm_core

@njit(fastmath=True, nopython=True)
def delta_2(dist_x, dist_y):
    dh_x = np.max(1 - dist_x, 0)
    dh_y = np.max(1 - dist_y, 0)

    return dh_x * dh_y

@njit(fastmath=True, nopython=True)
def delta_4(dist_x, dist_y):
    dh_x, dh_y = 0, 0
    val = 0

    #x-direction
    if (dist_x < 1):
        val = 1 + 4 * dist_x - 4 * (dist_x) ** 2

        if (val < 0):
            val = 0

        dh_x = 0.125 * (3.0 - 2.0 * dist_x + np.sqrt(val))
    elif (dist_x < 2):
        val = -7 + 12 * dist_x - 4 * (dist_x) ** 2

        if (val < 0):
            val = 0

        dh_x = 0.125 * (5.0 - 2.0 * dist_x - np.sqrt(val))
    else:
        dh_x = 0

    #y-direction
    if (dist_y < 1):
        val_y = 1 + 4 * dist_y - 4 * (dist_y) ** 2

        if (val_y < 0):
            val_y = 0

        dh_y = 0.125 * (3.0 - 2.0 * dist_y + np.sqrt(val_y))
    elif (dist_y < 2):
        val_y = -7 + 12 * dist_y - 4 * (dist_y) ** 2

        if (val_y < 0):
            val_y = 0

        dh_y = 0.125 * (5.0 - 2.0 * dist_y - np.sqrt(val_y))
    else:
        dh_y = 0

    return dh_x * dh_y

@njit(fastmath=True, nopython=True)
def interpolate_particle_velocities_kernel(
    NL, NX, x_b, y_b, x_start, x_end, y_start, y_end, ds,
    ux_grid, uy_grid, rho_grid, Uw_x, Uw_y, ux_b, uy_b, rho_b, Fx_IB, Fy_IB
):
    F_D = 0.0
    F_L = 0.0

    for i in range(NL):
        ux_b[i] = 0.0
        uy_b[i] = 0.0
        rho_b[i] = 0.0

        xb = x_b[i]
        yb = y_b[i]
        x_st = int(x_start[i])
        x_ed = int(x_end[i])
        y_st = int(y_start[i])
        y_ed = int(y_end[i])

        for x in range(x_st, x_ed + 1):
            xp = (x + NX) % NX

            for y in range(y_st, y_ed + 1):
                dist_x = geometry.periodic_dx(xb, xp)
                dist_y = abs(yb - y)

                # !CONTROL THE DELTA FUNCTION HERE!
                D = delta_4(dist_x, dist_y)

                ux_b[i] += ux_grid[xp, y] * D
                uy_b[i] += uy_grid[xp, y] * D
                rho_b[i] += rho_grid[xp, y] * D

        Fx_IB[i] = 2.0 * rho_b[i] * (Uw_x[i] - ux_b[i])
        Fy_IB[i] = 2.0 * rho_b[i] * (Uw_y[i] - uy_b[i])

        F_D += -Fx_IB[i] * ds[i]
        F_L += -Fy_IB[i] * ds[i]

    return F_D, F_L

def interpolate_particle_velocities():
    return interpolate_particle_velocities_kernel(
        initial.NL, initial.NX, geometry.particle.x_b, geometry.particle.y_b, geometry.particle.x_start, geometry.particle.x_end, geometry.particle.y_start, geometry.particle.y_end,
        geometry.particle.ds, lbm_core.ux, lbm_core.uy, lbm_core.rho, geometry.particle.Uw_x, geometry.particle.Uw_y,
        geometry.particle.ux_b, geometry.particle.uy_b, geometry.particle.rho_b, geometry.particle.Fx_IB, geometry.particle.Fy_IB
    )

@njit(fastmath=True, nopython=True)
def spread_particle_forces_kernel(
    NL, NX, NY, x_b, y_b, x_start, x_end, y_start, y_end, ds, Fx_IB, Fy_IB, fx, fy
):
    for x in range(NX):
        for y in range(NY):
            fx[x, y] = 0.0
            fy[x, y] = 0.0

    for n in range(NL):
        x_st = int(x_start[n])
        x_ed = int(x_end[n])
        y_st = int(y_start[n])
        y_ed = int(y_end[n])
        
        xb = x_b[n]
        yb = y_b[n]

        Fx_val = Fx_IB[n]
        Fy_val = Fy_IB[n]

        ds_n = ds[n]

        for x in range(x_st, x_ed + 1):
            xp = (x + NX) % NX

            for y in range(y_st, y_ed + 1):
                dist_x = geometry.periodic_dx(xb, xp)
                dist_y = abs(yb - y)

                D = delta_4(dist_x, dist_y)

                fx[xp, y] += Fx_val * D * ds_n
                fy[xp, y] += Fy_val * D * ds_n

def spread_particle_forces():
    p = geometry.particle
    spread_particle_forces_kernel(
        initial.NL, initial.NX, initial.NY, p.x_b, p.y_b, p.x_start, p.x_end,
        p.y_start, p.y_end, p.ds, p.Fx_IB, p.Fy_IB, lbm_core.fx, lbm_core.fy
    )

def update_particle_position():
    NX = initial.NX
    half_NX = NX * 0.5

    sum_fx = 0.0
    sum_fy = 0.0

    # Sum hydrodynamic forces acting on the particle across all markers
    for i in range(initial.NL):
        ds_i = geometry.particle.ds[i]
        sum_fx += -geometry.particle.Fx_IB[i] * ds_i
        sum_fy += -geometry.particle.Fy_IB[i] * ds_i

    # Linear momentum equation update (Center of Mass Velocities)
    M_s = initial.M_s
    M_f = initial.M_f
    rho0 = initial.rho_0
    rho_s = initial.rho_s

    alpha  = 0.7

    geometry.particle.Uc_next = geometry.particle.Uc_curr + (1 / M_s) * (sum_fx + initial.fx_ext * (M_s / rho_s)) + (M_f / M_s) * (geometry.particle.Uc_curr - geometry.particle.Uc_prev)
    geometry.particle.Vc_next = geometry.particle.Vc_curr + (1 / M_s) * (sum_fy + initial.fy_ext * (M_s / rho_s)) + (M_f / M_s) * (geometry.particle.Vc_curr - geometry.particle.Vc_prev)

    # Angular momentum equation update (Torque & Rotation)
    sum_torque_z = 0.0
    for i in range(initial.NL):
        arm_x = geometry.particle.x_b[i] - geometry.particle.x_centre
        if arm_x > half_NX:
            arm_x -= NX
        if arm_x < -half_NX:
            arm_x += NX
        arm_y = geometry.particle.y_b[i] - geometry.particle.y_centre

        F_on_particle_x = -geometry.particle.Fx_IB[i]
        F_on_particle_y = -geometry.particle.Fy_IB[i]

        torque_z = arm_x * F_on_particle_y - arm_y * F_on_particle_x
        sum_torque_z += torque_z * geometry.particle.ds[i]

    I_s = initial.I_s
    I_f = initial.I_f

    geometry.particle.omega_z_next = geometry.particle.omega_z_curr + (1 / I_s) * sum_torque_z + (I_f / I_s) * (geometry.particle.omega_z_curr - geometry.particle.omega_z_prev)
    
    # Spatial and rotational updates
    dx = 0.5 * (geometry.particle.Uc_curr + geometry.particle.Uc_next)
    dy = 0.5 * (geometry.particle.Vc_curr + geometry.particle.Vc_next)
    dphi = 0.5 * (geometry.particle.omega_z_curr + geometry.particle.omega_z_next)

    geometry.particle.total_x_distance += dx
    geometry.particle.x_centre += dx
    geometry.particle.x_centre = geometry.wrap(geometry.particle.x_centre)
    geometry.particle.y_centre += dy
    geometry.particle.phi += dphi

    # Save outputs to pass back to main orchestrator
    cx_out = geometry.particle.total_x_distance
    cy_out = geometry.particle.y_centre
    Uc_out = geometry.particle.Uc_curr
    Vc_out = geometry.particle.Vc_curr
    omega_z = geometry.particle.omega_z_curr

    # Shift velocity history time levels
    geometry.particle.Uc_prev = geometry.particle.Uc_curr
    geometry.particle.Uc_curr = geometry.particle.Uc_next
    geometry.particle.Vc_prev = geometry.particle.Vc_curr
    geometry.particle.Vc_curr = geometry.particle.Vc_next

    geometry.particle.omega_z_prev = geometry.particle.omega_z_curr
    geometry.particle.omega_z_curr = geometry.particle.omega_z_next

    # Re-generate markers for the new position and rotation angle phi
    geometry.marker()

    # Update marker surface velocities (Uw = Uc + omega x r)
    for i in range(initial.NL):
        arm_x = geometry.particle.x_b[i] - geometry.particle.x_centre
        if arm_x > half_NX:
            arm_x -= NX
        if arm_x < -half_NX:
            arm_x += NX
        arm_y = geometry.particle.y_b[i] - geometry.particle.y_centre

        geometry.particle.Uw_x[i] = geometry.particle.Uc_next - geometry.particle.omega_z_next * arm_y
        geometry.particle.Uw_y[i] = geometry.particle.Vc_next + geometry.particle.omega_z_next * arm_x

    return cx_out, cy_out, Uc_out, Vc_out, omega_z, sum_fy


def update_particle_position_diagnostic():
    # 1. Force particle velocities to exact zero (stationary body)
    geometry.particle.Uc_next = 0.00
    geometry.particle.Vc_next = 0.00
    geometry.particle.omega_z_next = 0.00

    # 2. Freeze center of mass position (no translation)
    cx_out = geometry.particle.total_x_distance
    cy_out = geometry.particle.y_centre
    Uc_out = 0.00
    Vc_out = 0.00

    # 3. Lock history time levels to zero
    geometry.particle.Uc_prev = 0.00
    geometry.particle.Uc_curr = 0.00
    geometry.particle.Vc_prev = 0.00
    geometry.particle.Vc_curr = 0.00
    geometry.particle.omega_z_prev = 0.00
    geometry.particle.omega_z_curr = 0.00

    # 4. Re-generate surface markers at fixed coordinates
    geometry.marker()

    # 5. Set target marker velocities to zero (non-moving surface)
    for i in range(initial.NL):
        geometry.particle.Uw_x[i] = 0.00
        geometry.particle.Uw_y[i] = 0.00

    return cx_out, cy_out, Uc_out, Vc_out
