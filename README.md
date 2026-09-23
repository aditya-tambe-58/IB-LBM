# Overview

The Python code in this repository simulates a particle suspended in 2D fluid flow, with a constant force in the x-direction. Fluid flow is simulated using the Lattice Boltzmann method, with the usual collision-streaming, but along with it, we use immersed boundary method, wherein Langragian markers are used to demarcate and enforce the boundary of the particle. Currently, 2D flow is implemented (implying a D2Q9 LBM scheme), and the suspended particle is an ellipse of controllable aspect ratio. 

# How to use it?

Initial conditions can be controlled in the initial.py file. Comments have been added to help identify the initial simulation settings. The delta function can be changed in lbm_coupler.py. To run the simulation, change the settings in initial.py and run main.py. The necessary libraries are mentioned in requirements.txt. 

# Next targets

A more detailed instruction doc, a visualization file, and in terms of physics, implementing 2 particles. 
