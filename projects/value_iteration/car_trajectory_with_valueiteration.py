# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:05:07 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
import argparse
###############################################################################
from pyro.dynamic  import vehicle
from pyro.planning import discretizer
from pyro.analysis import costfunction
from pyro.planning import valueiteration
from pyro.control  import controller
###############################################################################

sys  = vehicle.KinematicCarModelwithObstacles()

###############################################################################

# Planning

# Set domain
sys.x_ub = np.array([+35,+3.5,+3])
sys.x_lb = np.array([-5,-2,-3])

sys.u_ub = np.array( [+2,+0.3] )
sys.u_lb = np.array( [-2,-0.3] )

# Discrete world
grid_sys = discretizer.GridDynamicSystem( sys , (51, 51, 21) , (3,3) , 0.1 )
# Cost Function
cf = costfunction.QuadraticCostFunction(
    q=np.ones(sys.n),
    r=np.ones(sys.m),
    v=np.zeros(sys.p)
)

cf.xbar = np.array( [30, 0, 0] ) # target
cf.INF  = 1E8
cf.EPS  = 0.1
cf.R    = np.array([[0.1,0],[0,0]])

# VI algo

vi = valueiteration.ValueIteration_ND( grid_sys , cf )

vi.uselookuptable = True
vi.initialize()
#if load_data:
# vi.load_data('car_vi_2')
vi.compute_steps(100, plot=True, maxJ=6000)
#if save_data:
vi.save_data('car_vi_2')

vi.assign_interpol_controller()

vi.plot_cost2go(maxJ=6000)
vi.plot_policy(0)
vi.plot_policy(1)

cl_sys = controller.ClosedLoopSystem( sys , vi.ctl )
#
## Simulation and animation
x0   = [0, 0 ,0]
tf   = 10

sim = cl_sys.compute_trajectory( x0 , tf , 10001)
cl_sys.get_plotter().plot(sim, 'xu')
cl_sys.get_animator().animate_simulation(sim, save=True, file_name='car_2')