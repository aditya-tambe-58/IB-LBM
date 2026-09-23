# The main file that runs the collision-streaming loop while maintaining the Langragian markers.
# Author: Aditya Tambe

import os
import time
import initial
import lbm_core
import lbm_coupler
from tqdm import tqdm
import sys

def save_case_dir(case_dir):
    """Creates the output directory for simulation data if it doesn't exist."""
    if case_dir and not os.path.exists(case_dir):
        os.makedirs(case_dir)

import sys
import time

def print_progress_bar(step, total_steps, start_time, bar_len=40, update_interval=1000):
    """
    Renders a 2-line minimalist progress bar with a horizontal line fill:
    Line 1: IB-LBM Simulation  70%  (ETA: 04:29)
    Line 2:   |━━━━━━━━━━━━━━━━━━━━                  |
    """
    # Throttle terminal updates to keep execution speed high
    if step % update_interval != 0 and step != total_steps and step != total_steps - 1:
        return

    # Cyan/Blue ANSI color matching the reference palette
    COLOR = "\033[38;2;30;160;220m"
    RESET = "\033[0m"

    elapsed = time.time() - start_time
    percent = (step / total_steps) * 100
    filled_len = int(bar_len * step // total_steps)

    # ETA Calculation
    speed = step / elapsed if elapsed > 0 else 0
    remaining_sec = (total_steps - step) / speed if speed > 0 else 0
    mins, secs = divmod(int(remaining_sec), 60)
    hrs, mins = divmod(mins, 60)
    eta_str = f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0 else f"{mins:02d}:{secs:02d}"

    # Overwrite the previous 2 lines in-place
    if getattr(print_progress_bar, "has_run", False):
        sys.stdout.write("\033[2A\r\033[K")
    else:
        print_progress_bar.has_run = True

    # Heavy horizontal line '━' fill (use '─' for a thinner line)
    bar = "━" * filled_len + " " * (bar_len - filled_len)

    # Line 1: Status & Percentage
    sys.stdout.write(f"IB-LBM Simulation  {percent:3.0f}%  (ETA: {eta_str})\033[K\n")
    # Line 2: Colored pipes and line fill
    sys.stdout.write(f"  {COLOR}|{bar}|{RESET}\033[K\n")
    sys.stdout.flush()

    # Reset toggle on final iteration
    if step >= total_steps - 1:
        print_progress_bar.has_run = False

def main():
    start_time = time.time()
    
    count = 0
    case_dir = "simulation_results"
    save_case_dir(case_dir)

    # Print terminal tracking info, initialize grid/particles, and save parameters
    lbm_core.print_simulation_info()
    lbm_core.initialize_simulation()
    lbm_core.write_simulation_parameters()

    # Format force and trajectory filenames using case directory and parameters
    re_num = getattr(initial, 'Re', 100)
    force_filename = os.path.join(case_dir, f"forces_Re{re_num}_x0{initial.X0}.dat")
    traj_filename = os.path.join(case_dir, f"trajectory_Re{re_num}_x0{initial.X0}.dat")

    print("\nStarting simulation time loop...")
    
    with open(force_filename, "w") as f_force, open(traj_filename, "w") as f_traj:
        # Write Tecplot/ASCII header formats
        f_force.write('VARIABLES = "Time", "CD", "CL"\n')
        f_traj.write('VARIABLES = "Step", "Time_Star", "X_Star", "Y_Star", "U_Center", "V_Center", "Omega_z"\n')

        progress_interval = max(1, initial.t_max // 20)

        for k in range(initial.t_max):
            # Print a progress dot at each 5% interval
            print_progress_bar(k, initial.t_max, start_time, bar_len=40, update_interval=1000)

            # 1. Compute macroscopic fluid observations from current distributions
            lbm_core.cal_obs()

            # 2. Save full grid data snapshot halfway through simulation
            if k == initial.t_max // 2 and k > 0:
                lbm_core.data(count, phase_id=1, case_dir=case_dir)
                count += 1

            # 3. Interpolate fluid velocities at markers and get hydrodynamic forces
            # (Comment out the below two lines for removing the partice.)
            cd, cl = lbm_coupler.interpolate_particle_velocities()
            f_force.write(f"{k} {cd:.6f} {cl:.6f}\n")

            # 4. Spread particle forces to grid, execute BGK collision, and stream
            # (Comment out the below line to remove the particle.)
            lbm_coupler.spread_particle_forces()
            lbm_core.collision()
            lbm_core.streaming()

            # 5. Update particle position and velocities, then write to trajectory file
            # (Comment out the below line for removing the particle.)
            cx_out, cy_out, Uc_out, Vc_out, omega_z, sum_fy = lbm_coupler.update_particle_position()

            # Calculate dimensionless time and normalized spatial coordinates
            # (Comment out the below line for removing the particle.)
            t_star = (k * initial.u_mean) / initial.H
            x_star = cx_out / initial.H
            y_star = cy_out / initial.H

            # Log a single, unified row matching the header columns
            # (Comment out the below line for removing the particle.)
            # Change from :.6f to :.6e
            f_traj.write(f"{k} {t_star:.6f} {x_star:.6f} {y_star:.6f} {Uc_out:.6e} {Vc_out:.6e} {omega_z:.6e}\n")

    print("\nSimulation time loop completed successfully!")

    # Save final flow snapshot, dump restart state, and output total execution runtime
    lbm_core.data(count, phase_id=2, case_dir=case_dir)
    lbm_core.restartrum()
    lbm_core.print_runtime(start_time, initial.t_max)

if __name__ == "__main__":
    main()
