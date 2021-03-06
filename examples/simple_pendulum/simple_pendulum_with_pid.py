# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:05:08 2018

@author: Alexandre
"""
###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic  import pendulum
from pyro.control  import robotcontrollers
from pyro.analysis import simulation
###############################################################################

sys  = pendulum.SinglePendulum()

dof = 1

kp = 2 # 2,4
kd = 1 # 1
ki = 1

ctl  = robotcontrollers.JointPID( dof, kp , ki, kd)

# Set Point
q_target = np.array([3.14])
ctl.rbar = q_target

# New cl-dynamic
cl_sys = ctl + sys

# Simultation
x_start  = np.array([0,0])

cl_sys.sim = simulation.CLosedLoopSimulation( cl_sys , 20 , 20001 , 'euler' )
cl_sys.sim.x0 = x_start
cl_sys.sim.compute()
cl_sys.sim.phase_plane_trajectory(0,1)
cl_sys.sim.plot('xu')
cl_sys.animate_simulation()