"""
Course code: CS 489
Course title: Introduction to Unmanned Aerial Systems
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
    Implementation of cascade PID controllers for position control of quadrotor

"""

import numpy as np

class PID:
    def __init__(self, kp, ki, kd):
        # Gains
        self._kp = kp
        self._ki = ki
        self._kd = kd

        # Integral term
        self._int = 0
        # Maximum PID output (to avoid integral wind up)
        self._maxInt = 0

        # Last error value (used for the derivative term)
        self._last_e = 0

        # PID output
        self._u = 0

    def setMaxInt(self, val):
        self._maxInt = val

    def update(self, e, dt):
        u = self._kp*e +  (e - self._last_e)/dt
        self._last_e = e

        self._int += self._ki*e*dt
        if abs(self._int) < self._maxInt:
            u = u + self._int

        return u

    def resetPID(self):
        self._int = 0
        self._last_e = 0

    def setPID(self, p,i,d):
        self._kp = p
        self._ki = i
        self._kd = d

class Controller:
    def __init__(self):
        # XY PIDs
        self._x_pid = PID(1.0, 0,0)
        self._y_pid = PID(1.0, 0,0)
        self._vx_pid = PID(1.0, 0.2, 0.003)
        self._vy_pid = PID(1.0, 0.2, 0.003)
        # Z PID
        self._z_pid = PID(1.0, 0,0)
        self._vz_pid = PID(2.0, 0.2, 0.003)

        # roll/pitch/yaw PID
        self._roll_pid = PID(0.6, 0, 0)
        self._pitch_pid = PID(0.6, 0, 0)
        self._yaw_pid = PID(0.5,0,0)

        # Roll/pitch/yaw rate PID
        self._roll_rate_pid = PID(0.2, 0.2, 0.003)
        self._pitch_rate_pid = PID(0.2, 0.2, 0.003)
        self._yaw_rate_pid = PID(0.3, 0.2, 0.)

        # Maximum XY velocity, m/s
        self._maxXYVel = 5.0
        # Maximum Z velocity, m/s
        self._maxZVel = 2.0

        # Maximum tilt angles
        self._maxTilt = 45*np.pi/180.
        # Maximum angular rate, rad/s
        self._maxAngRate = 200*np.pi/180.