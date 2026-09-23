# The module that is the core engine of the LBM simulation. It handles the fluid particle streaming and collision.  
# Author: Aditya Tambe

import os
import time
import numpy as np
from numba import njit, prange
import initial
import geometry

# Precomputed periodic wrapping tables
next_x = np.zeros((initial.NX, initial.Q), dtype=np.int32)
next_y = np.zeros((initial.NY, initial.Q), dtype=np.int32)

# Distribution function arrays (Current and Temporary)
f = np.zeros((initial.NX, initial.NY, initial.Q), dtype=np.float64)
f_temp = np.zeros((initial.NX, initial.NY, initial.Q), dtype=np.float64)

# Macroscopic fluid property variables
rho = np.zeros((initial.NX, initial.NY), dtype=np.float64)      
ux = np.zeros((initial.NX, initial.NY), dtype=np.float64)       
uy = np.zeros((initial.NX, initial.NY), dtype=np.float64)       
ux_eq = np.zeros((initial.NX, initial.NY), dtype=np.float64)    # Forced x-velocity (Guo's approx)
uy_eq = np.zeros((initial.NX, initial.NY), dtype=np.float64)    # Forced y-velocity (Guo's approx)

# Boundary flags and force density grids
bnode = np.zeros((initial.NX, initial.NY), dtype=np.float64)    # Boundary nodes (1 at walls, 0 elsewhere)
fx = np.zeros((initial.NX, initial.NY), dtype=np.float64)       # Force density in x-direction
fy = np.zeros((initial.NX, initial.NY), dtype=np.float64)       # Force density in y-direction

@njit(fastmath=True, parallel=True, nopython=True)
def tables_kernel(NX, NY, Q, cx, cy, next_x, next_y):
    for x in prange(NX):
        for y in range(NY):
            for i in range(Q):
                next_x[x, i] = (x + cx[i] + NX) % NX
                next_y[y, i] = (y + cy[i] + NY) % NY

def tables():
    tables_kernel(
        initial.NX, initial.NY, initial.Q, 
        initial.cx, initial.cy, next_x, next_y
    )

def readrestart():
    try:
        with open("restart", "r") as f4:
            for x in range(initial.NX):
                for y in range(initial.NY):
                    for i in range(initial.Q):
                        line = f4.readline()
                        if not line:
                            raise ValueError("Unexpected end of restart file!")
                        f[x, y, i] = float(line.strip())
    except FileNotFoundError:
        print("Error: Could not open restart file!")
        exit(1)

    # Assign boundary nodes (0 for fluid, 1 for walls at y = 0 and y = NY-1)
    for x in range(initial.NX):
        for y in range(initial.NY):
            bnode[x, y] = 0.0
            if y == 0 or y == initial.NY - 1:
                bnode[x, y] = 1.0

def initialise():
    for x in range(initial.NX):
        for y in range(initial.NY):
            rho[x, y] = 1
            ux[x, y] = 0
            uy[x, y] = 0

            for i in range(initial.Q):
                f[x, y, i] = rho[x, y] * initial.t[i]

            bnode[x, y] = 0
            if(y == 0 or y == initial.NY - 1):
                bnode[x, y] = 1


    for i in range(initial.NL):
        geometry.particle.Uw_x[i] = 0
        geometry.particle.Uw_y[i] = 0

@njit(fastmath=True, parallel=True, nopython=True)
def cal_obs_kernel(NX, NY, Q, f_arr, rho_arr, ux_arr, uy_arr, cx, cy):
    for x in prange(NX):
        for y in range(NY):
            r_val = 0.0
            for i in range(Q):
                r_val += f_arr[x, y, i]
            rho_arr[x, y] = r_val

            u_val = 0.0
            v_val = 0.0
            for i in range(Q):
                u_val += cx[i] * f_arr[x, y, i]
                v_val += cy[i] * f_arr[x, y, i]

            ux_arr[x, y] = u_val / r_val
            uy_arr[x, y] = v_val / r_val

def cal_obs():
    cal_obs_kernel(initial.NX, initial.NY, initial.Q, f, rho, ux, uy, initial.cx, initial.cy)

@njit(fastmath=True, parallel=True, nopython=True)
def collision_kernel(
    NX, NY, Q, fx_ext, fy_ext, omega, cx, cy, t,
    bnode, fx, fy, rho, ux, uy, ux_eq, uy_eq, f
):
    for x in prange(NX):
        for y in range(NY):
            if bnode[x, y] != 1.0:
                Fx_total = fx[x, y] + fx_ext
                Fy_total = fy[x, y] + fy_ext
                
                # Guo's velocity formulation
                ux_eq_val = ux[x, y] + (0.5 * Fx_total) / rho[x, y]
                uy_eq_val = uy[x, y] + (0.5 * Fy_total) / rho[x, y]
                
                ux_eq[x, y] = ux_eq_val
                uy_eq[x, y] = uy_eq_val
                
                u_squared = ux_eq_val ** 2 + uy_eq_val ** 2

                for i in range(Q):
                    udotc = ux_eq_val * cx[i] + uy_eq_val * cy[i]
                    
                    # Equilibrium distribution function
                    feq = rho[x, y] * t[i] * (
                        1.0 + 3.0 * udotc + 4.5 * udotc * udotc - 1.5 * u_squared
                    )

                    # Guo's discrete forcing terms
                    term1 = (3.0 * (cx[i] - ux_eq_val) + 9.0 * cx[i] * udotc) * Fx_total
                    term2 = (3.0 * (cy[i] - uy_eq_val) + 9.0 * cy[i] * udotc) * Fy_total
                    force = (1.0 - 0.5 * omega) * t[i] * (term1 + term2)

                    # BGK collision update with Guo forcing
                    f[x, y, i] = f[x, y, i] - omega * (f[x, y, i] - feq) + force

def collision():
    collision_kernel(
        initial.NX, initial.NY, initial.Q, initial.fx_ext, initial.fy_ext,
        initial.omega, initial.cx, initial.cy, initial.t,
        bnode, fx, fy, rho, ux, uy, ux_eq, uy_eq, f
    )

@njit(fastmath=True, parallel=True, nopython=True)
def streaming_kernel(NX, NY, Q, bnode, f, f_temp, next_x, next_y, opp):
    # 1. Store post-collision distribution function into f_temp
    for x in prange(NX):
        for y in range(NY):
            if bnode[x, y] != 1.0:
                for i in range(Q):
                    f_temp[x, y, i] = f[x, y, i]

    # 2. Standard streaming (Push populations from fluid nodes to target nodes)
    for x in prange(NX):
        for y in range(NY):
            if bnode[x, y] != 1.0:
                for i in range(Q):
                    newx = next_x[x, i]
                    newy = next_y[y, i]
                    f[newx, newy, i] = f_temp[x, y, i]

    # 3. Half-way bounce-back scheme (Target node is a solid wall)
    for x in prange(NX):
        for y in range(NY):
            if bnode[x, y] != 1.0:
                for i in range(Q):
                    newx = next_x[x, i]
                    newy = next_y[y, i]
                    
                    if bnode[newx, newy] == 1.0:
                        f[x, y, opp[i]] = f_temp[x, y, i]

def streaming():
    streaming_kernel(
        initial.NX, initial.NY, initial.Q,
        bnode, f, f_temp, next_x, next_y, initial.opp
    )

def initialize_simulation():
    print("Initializing...")

    # Precompute LBM lattice velocity tables
    tables()

    #Create a particle.
    geometry.init_particle(initial.X0, initial.Y0, initial.NL)

    # Check if restart file exists (equivalent to C's stat("restart", &info))
    if os.path.exists("restart"):
        print("Loading restart file.")
        readrestart()
    else:
        print("Running fresh start.")
        initialise()

    # Initialize particle position and tracking variables
    geometry.particle.x_centre = initial.X0
    geometry.particle.y_centre = initial.Y0
    geometry.particle.total_x_distance = initial.X0  # Tracks total x-displacement for periodic boundary conditions

    # Initialize center-of-mass linear velocities (prev, curr, next time levels)
    geometry.particle.Uc_prev = 0.0
    geometry.particle.Uc_curr = 0.0
    geometry.particle.Uc_next = 0.0

    geometry.particle.Vc_prev = 0.0
    geometry.particle.Vc_curr = 0.0
    geometry.particle.Vc_next = 0.0

    # Initialize center-of-mass angular velocity
    geometry.particle.omega_z_prev = 0.0
    geometry.particle.omega_z_curr = 0.0
    geometry.particle.omega_z_next = 0.0

    # Generate initial Lagrangian markers for the ellipse
    geometry.marker()

#Tracking functions, written by AI

def print_simulation_info():
    """Displays core simulation specifications and geometry details to the console."""
    print("==================================================")
    print("      LBM-IBM 2D Ellipse Simulation (Python)      ")
    print("==================================================")
    print(f"Domain Size (NX x NY)  : {initial.NX} x {initial.NY}")
    print(f"Reynolds Number (Re)   : {getattr(initial, 'Re', 'N/A')}")
    print(f"Minor Axis (b)         : {initial.minor_axis}")
    print(f"Aspect Ratio (AR)      : {initial.aspect_ratio}")
    print(f"Initial Position (X0,Y): ({initial.X0}, {initial.Y0})")
    print(f"Marker Count (NL)      : {initial.NL}")
    print(f"Total Timesteps        : {initial.t_max}")
    print(f"Relaxation Parameter   : {initial.omega}")
    print(f"External Force (fx)    : {initial.fx_ext}")
    
    # Estimate memory footprint of core arrays
    mem_bytes = f.nbytes + f_temp.nbytes + rho.nbytes + ux.nbytes + uy.nbytes
    print(f"Estimated Grid Memory  : {mem_bytes / (1024 * 1024):.2f} MB")
    print("Execution Backend      : Python with Numba JIT")
    print("==================================================")


def print_runtime(start_time, current_step):
    """Calculates and displays elapsed wall time and simulation speed."""
    elapsed = time.time() - start_time
    minutes = elapsed / 60.0
    speed = current_step / elapsed if elapsed > 0 else 0.0
    print(f"Runtime: {elapsed:.2f} seconds ({minutes:.2f} minutes) | Speed: {speed:.2f} steps/sec")


def data(data_counter, phase_id, case_dir="."):
    """Writes the flow field data to a Tecplot-compatible ASCII file."""
    os.makedirs(case_dir, exist_ok=True)
    filename = os.path.join(case_dir, f"flow_field_{phase_id}_{data_counter:05d}.dat")
    
    with open(filename, "w") as f_out:
        f_out.write('TITLE = "LBM Flow Field - Ellipse"\n')
        f_out.write('VARIABLES = "X", "Y", "RHO", "U", "V"\n')
        f_out.write(f'ZONE I={initial.NX}, J={initial.NY}, DATAPACKING=POINT\n')
        
        for y in range(initial.NY - 1, -1, -1):
            for x in range(initial.NX):
                f_out.write(f"{x} {y} {rho[x, y]:.6f} {ux_eq[x, y]:.6f} {uy_eq[x, y]:.6f}\n")


def write_simulation_parameters():
    """Logs simulation settings and particle geometry specs to a text summary file."""
    filename = "simulation_parameters.txt"
    try:
        with open(filename, "w") as f:
            f.write("=========================================\n")
            f.write("      SIMULATION PARAMETERS SUMMARY       \n")
            f.write("=========================================\n")
            f.write(f"Domain Dimensions (NX x NY) : {initial.NX} x {initial.NY}\n")
            f.write(f"Minor Axis (b)              : {initial.minor_axis}\n")
            f.write(f"Aspect Ratio (AR)           : {initial.aspect_ratio}\n")
            f.write(f"Initial Center (X0, Y0)     : {initial.X0}, {initial.Y0}\n")
            f.write(f"Lagrangian Markers (NL)     : {initial.NL}\n")
            f.write(f"Reynolds Number (Re)        : {getattr(initial, 'Re', 'N/A')}\n")
            f.write(f"Relaxation Time (omega)     : {initial.omega}\n")
            f.write(f"Kinematic Viscosity         : {getattr(initial, 'nu', 'N/A')}\n")
            f.write(f"External Forcing (fx, fy)   : {initial.fx_ext}, {initial.fy_ext}\n")
            f.write(f"Total Timesteps             : {initial.t_max}\n")
            f.write("=========================================\n")
    except IOError:
        print("Error: Could not write simulation_parameters.txt file!")


def restartrum():
    """Dumps the current distribution functions into a restart file."""
    try:
        with open("restart", "w") as f_out:
            for x in range(initial.NX):
                for y in range(initial.NY):
                    for i in range(initial.Q):
                        f_out.write(f"{f[x, y, i]}\n")
    except IOError:
        print("Error: Could not write restart file!")


