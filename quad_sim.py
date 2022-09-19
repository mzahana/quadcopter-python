"""
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
    Implementation of cascade PID controllers for position control of quadrotor

"""
import time
import numpy as np
from quadcopter import Quadcopter
from controller import Controller
from quad_dynamics import Dynamics
from draw_quad import  QuadPlot
import signal
import sys
import argparse
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt

run=True

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)

# Quadcopter object
q=Quadcopter()
# Dynamics object
quad_dyn = Dynamics(q)
# Controller object
params={'xy_kp': 1.0, 'z_kp': 1.0,
    'vxy_pid':[1.0,0.2, 0.0004], 'vz_pid': [1.0, .02, 0.0003],
    'roll_kp': 6.0, 'pitch_kp': 6.0, 'yaw_kp': 2.0,
    'roll_speed_pid': [0.1, .200, 0.0003],
    'pitch_speed_pid': [0.1, .200, 0.0003], 
    'yaw_speed_pid': [0.1, 0.2, 0.0]}
cont = Controller(q,params)
cont.setDesiredPos(np.array([0,0,1]))
cont.setDesiredYaw(45*np.pi/180.)

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
quad_plt = QuadPlot(ax, q)

# Catch Ctrl+C to stop threads
signal.signal(signal.SIGINT, signal_handler)

# Start threads
quad_dyn.startThread(dt=0.002)
cont.startThread(update_rate=0.005)
while(run==True):
    quad_plt.update()
    # time.sleep(0.1)
    print(q.getPosition())
    # plt.pause(0.000000000000001)

quad_dyn.stopThread()
cont.stopThread()