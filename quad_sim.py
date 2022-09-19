"""
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
    Implementation of cascade PID controllers for position control of quadrotor

"""
from controller import Controller
from quad_dynamics import Dynamics
from draw_quad import  QuadPlot
import signal
import sys
import argparse
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt

run=True

# Dynamics object
quad_dyn = Dynamics()
# Controller
params={'xy_kp': 1.0, 'z_kp': 1.0,
    'vxy_pid':[1.0,0.2, 0.004], 'vz_pid': [1.0, .02, 0.003],
    'roll_kp': 6.0, 'pitch_kp': 6.0, 'yaw_kp': 2.0,
    'roll_speed_pid': [0.15, 0.2, 0.003],
    'pitch_speed_pid': [0.15, 0.2, 0.003], 
    'yaw_speed_pid': [0.15, 0.2, 0.0]}
cont = Controller(params)

# TODO: we need some share object between the Controller and Quadcopter dynamics!!!!

# Quad plotter
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# Define World frame
# origin
O_W = [0,0,0]
# translation
W = SE3.Trans(O_W)
# Plot world frame
W.plot( frame='W', color="green", length=3, axes=ax)
quad_plt = QuadPlot(arm_l=0.22*2, prop_l=0.2286, ax=ax)

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)