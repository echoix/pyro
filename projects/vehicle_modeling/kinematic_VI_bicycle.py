# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:01:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import vehicle
from pyro.control  import linear
import matplotlib.pyplot as plt

###############################################################################

# "Fake" controller - Varying inputs (delta, Vx) throughout time (change in linear.py)
ctl = linear.kinematicInputs()

# Vehicule dynamical system
sys = vehicle.KinematicBicyleModel()

# Set default wheel velocity and steering angle
cl_sys = ctl+sys


# Plot open-loop behavior (ex: np.array[intial_conditions], time_of_simulation)
cl_sys.plot_trajectory( np.array([0,0,0]) , 10)

# Rebuild x,u and t from simulation
x = cl_sys.sim.x_sol
u = cl_sys.sim.u_sol 
t = cl_sys.sim.t   

# Plot the vehicle's trajectory
figsize   = (7, 4)
dpi       = 100
plt.figure(2,figsize=figsize, dpi=dpi)
plt.axes(xlim=(-20,150), ylim=(-5,200))  
plt.plot(x[:,0],x[:,1],'--g', label = 'Kinematic Model')     
plt.legend(fontsize ='15')   
plt.legend(fontsize ='15')
plt.title("Vehicle's cartesian position", fontsize=20)
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.show()

# Animate the simulation
cl_sys.animate_simulation()