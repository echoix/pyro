# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:28:17 2018

@author: Alexandre
"""

###############################################################################
import numpy as np
###############################################################################
from pyro.dynamic import pendulum
from pyro.planning import randomtree
###############################################################################

sys  = pendulum.DoublePendulum()

x_start = np.array([-3.14,0,0,0])
x_goal  = np.array([0,0,0,0])

planner = randomtree.RRT( sys , x_start )

t = 10
    
planner.u_options = [
        np.array([-t,-t]),
        np.array([-t,+t]),
        np.array([+t,-t]),
        np.array([+t,+t]),
        np.array([ 0,+t]),
        np.array([ 0,-t]),
        np.array([ 0, 0]),
        np.array([+t, 0]),
        np.array([-t, 0])
        ]

planner.goal_radius = 0.8

planner.find_path_to_goal( x_goal )

planner.plot_tree()
planner.plot_open_loop_solution()
sys.animate_simulation()