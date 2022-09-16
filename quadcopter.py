"""
Course code: CS 489
Course title: Introduction to nmanned Aerial Systems
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
Visualize 2D and 3D coordinate frames

Reference videos
- https://youtu.be/c8Xb7krH2_w 
"""

import numpy as np
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt
import scipy.integrate
import time
import datetime
import threading

class Quadcopter:
    def __init__(self, weight=1.0, L=.3, r=0.1) :
        self._kf = 0.228 # motor force constant
        self._km = 0.12 # motor torque constant
        self._gamma = self._km/self._kf
        self._m = weight
        self._L = L # arm length, half of total body length
        self._r = r # radius of sphere that is used to estimate inertia
        
        # Inertia is estimated according to the following reference
        # Quadrotor Dynamics and Control Rev , Randal Beard
        # https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=2324&context=facpub
        ixx = (2*self._m*self._r**2/5) + (2*self._L**2/self._m)
        iyy = self._ixx
        izz = (2*self._m*self._r**2/5) + (4*self._L**2/self._m)
        self._I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self._Iinv = np.linalg.inv(I)

        self._g = 9.18 # gravity m/s/s

        # Current state
        # x= [x y z x_dot y_dot z_dot phi theta psi phi_dot theta_dot psi_dot]
        self._state = np.zeros(12)

        # Rotation matrix
        self._R = np.eye(3,3)

        # Inputs
        # u_1 = total body thrust, (float), Newton
        self._u1 = 0
        # Moments u2=[]ux,uy,uz
        self._u2 = np.array([0,0,0])

        # Control allocation matrix
        # Transforms motor forces to thrust and moments
        self._C = np.array([[1,1,1,1],
                            [0,0,self._L, -self._L],
                            [-self._L, self._L, 0,0],
                            [-self._gamma, -self._gamma,self._gamma,self._gamma]])
        # Transforms thrust and moments to motor forces
        self._Cinv = np.linalg.inv(self._C)

        self._motor_speeds = np.zeros(4)

        self._ode =  scipy.integrate.ode(self.stateDot).set_integrator('vode',nsteps=500,method='bdf')

        self._time = datetime.datetime.now()

        self._thread_object = None

        self._run = True


    def wrapAngle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def inputsFromMotorSpeeds(self,w):
        """
        Calculate inputs self._u1 from motors speeds

        Params:
        w: (4x1 np.ndarray) motors speeds
        """
        u = self._C.dot(self._kf*w**2)
        self._u1 = u[0]
        self._u2 = u[1:]

    def motorSpeedsFromInputs(self,u):
        """
        Calculate motor speeds from inputs self._u1, self._u2

        Params:
        u: (4x1 np.ndarray) u1(1x1)=thrust, u2(3x1)=moments
        """
        F=self._Cinv.dot(u)
        self._motor_speeds = np.sqrt(F/self._kf)

    def clacR(self, rpy):
        """
        Calculate roation marix self._R from Euler angles
        Uses ZYX order
        
        Params:
        rpy: 3x1 np.ndarray roll, pitch, yaw in radians
        """
        R = SO3.Rz(rpy[2])*SO3.Ry(rpy[1])*SO3.Rx(rpy[0])
        self._R = R.R

    def stateDot(self, time, state):
        # States, x= [x y z x_dot y_dot z_dot phi theta psi phi_dot theta_dot psi_dot]
        x_dot = np.zeros(12)

        # Linear velocities
        x_dot[0] = state[3]
        x_dot[1] = state[4]
        x_dot[2] = state[5]
        # Linear accelerations
        self.clacR(state[6:9])
        self.inputsFromMotorSpeeds(self._motor_speeds) # calculate self._u1, self._u2
        acc = np.array([0,0, -self._g]) + (1/self._m) * self._R.dot(np.array([0,0,self._u1]))
        x_dot[3] = acc[0]
        x_dot[4] = acc[1]
        x_dot[5] = acc[2]
        # angular rates
        x_dot[6] = state[9]
        x_dot[7] = state[10]
        x_dot[8] = state[11]
        # Angular acceleration
        omega = state[9:12]
        ommega_dot = np.dot(self._Iinv, ( self._u2- np.cross(omega, np.dot(self._I, omega)) ) )
        x_dot[9] = ommega_dot[0]
        x_dot[10] = ommega_dot[1]
        x_dot[11] = ommega_dot[2]

        return x_dot

    def update(self, dt):
        self._ode.set_initial_value(self._state,0)
        self._state = self._ode.integrate(self._ode.t + dt)
        self._state[6:9] = self.wrapAngle(self._state[6:9])
        # self._state[2] = max(0,self._state[2])

    def setRPY(self, rpy):
        """
        Sets Euler angles, roll, pitch, yaw

        Params:
        rpy: 3x1 np.ndarray roll pitch yaw angles in radian
        """
        self._state[6:9] = rpy
    
    def setPosition(self, xyz):
        """
        Sets position

        Params:
        rpy: 3x1 np.ndarray [x,y,z]
        """
        self._state[0:3] = xyz

    def setMotorSpeeds(self, w):
        self._motor_speeds=w

    def getPosition(self):
        return self._state[0:3]

    def getSpeed(self):
        return self._state[3:6]

    def getRPY(self):
        return self._state[6:9]

    def getAngularSpeed(self):
        return self._state[9:12]
        
    def threadRun(self,dt):
        rate = dt
        last_update = self._time
        while(self._run==True):
            time.sleep(0)
            self._time = datetime.datetime.now()
            if (self._time-last_update).total_seconds() > rate:
                self.update(dt)
                last_update = self._time

    def startThread(self,dt=0.002):
        self._thread_object = threading.Thread(target=self.threadRun,args=(dt))
        self._thread_object.start()

    def stopThread(self):
        self._run = False