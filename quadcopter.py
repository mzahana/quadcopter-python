"""
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
Implementation of quadrotor dynamics

"""

import numpy as np
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt
import scipy.integrate
import time
import datetime
import threading

class Quadcopter:
    def __init__(self, weight=2.0, L=.22, r=0.1, max_m_rpm=13000, kf=1.4e-5, km=2.0e-6) :
        """
        Params:
            weight: (float) wuadcopter weight in Kg
            L: (float) Arm length - half body length, in meters
            r: (float) adius of sphere that is used to estimate inertia
            kf: (float) Motor's force constant
            km: (float) Motor's torque constant
            max_m_rpm:R (int) Maximum motor RPM
        """
        self._kf = kf
        self._km = km
        self._gamma = self._km/self._kf
        self._m = weight
        self._L = L # arm length, half of total body length
        self._r = r # radius of sphere that is used to estimate inertia

        self._max_motor_rpm = max_m_rpm
        self._max_motor_rps = self._max_motor_rpm * 0.10472 # in rad/sec
        
        # Inertia is estimated according to the following reference
        # Quadrotor Dynamics and Control Rev , Randal Beard
        # https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=2324&context=facpub
        ixx = (2*self._m*self._r**2/5) + (2*self._L**2/self._m)
        iyy = ixx
        izz = (2*self._m*self._r**2/5) + (4*self._L**2/self._m)
        self._I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self._Iinv = np.linalg.inv(self._I)

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
        Calculate motor speeds from inputs u=[self._u1, self._u2]

        Params:
            u: (4x1 np.ndarray) u=[u1(1x1)=thrust, u2(3x1)=moments]
        """
        F=self._Cinv.dot(u)
        self._motor_speeds = np.sqrt(F/self._kf)
        np.clip(self._motor_speeds, 0, self._max_motor_rps)

    def clacR(self, rpy):
        """
        Calculates roation marix self._R from Euler angles
        Uses ZYX order
        
        Params:
            rpy: 3x1 np.ndarray roll, pitch, yaw in radians
        """
        R = SO3.Rz(rpy[2])*SO3.Ry(rpy[1])*SO3.Rx(rpy[0])
        self._R = R.R

    def getR(self):
        """
        Returns rotation matrix of current attitude
        """
        # x= [x y z x_dot y_dot z_dot phi theta psi phi_dot theta_dot psi_dot]
        rpy=np.zeros(3)
        rpy[0]=self._state[6]
        rpy[1]=self._state[7]
        rpy[2]=self._state[8]
        R = SO3.Rz(rpy[2])*SO3.Ry(rpy[1])*SO3.Rx(rpy[0])
        return R.R

    def calcRPYdot(self, rpy, pqr):
        """
        Transforms body angular speed (p,q,r) to Euler angles rates (roll_rate, pitch_rate, yaw_rate)

        Params:
            rpy: 3x1 roll, pitch, yaw
            pqr: 3x1 body angular rates
        Returns:
            rpy_dot: 3x1 Euler angles rates
        """
        r = rpy[0]
        p = rpy[1]
        R = np.array([ [np.cos(p), 0, np.sin(p)],
                        [np.sin(p)*np.tan(r), 1, -np.cos(p)*np.tan(r)],
                        [-np.sin(p)/np.cos(r), 0, np.cos(p)/np.cos(r)] ])
        rpy_dot = np.dot(R,pqr)
        return rpy_dot

    def setState(self, state):
        self._state = state

    def stateDot(self, time, state):
        #            [0 1 2  3     4      5    6    7    8  9 10 11]
        # States, x= [x y z x_dot y_dot z_dot phi theta psi p q r]
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
        rpy_dot = self.calcRPYdot(self._state[6:9], self._state[9:12])
        x_dot[6] = rpy_dot[0]
        x_dot[7] = rpy_dot[1]
        x_dot[8] = rpy_dot[2]
        # Angular acceleration
        omega = state[9:12]
        omega_dot = np.dot(self._Iinv, ( self._u2- np.cross(omega, np.dot(self._I, omega)) ) )
        x_dot[9] = omega_dot[0]
        x_dot[10] = omega_dot[1]
        x_dot[11] = omega_dot[2]

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
        self._thread_object = threading.Thread(target=self.threadRun,args=(dt,))
        self._thread_object.start()

    def stopThread(self):
        self._run = False

if __name__=="__main__":
    quad = Quadcopter()
    init_state = np.zeros(12)
    quad.setState(init_state)
    u = np.array([1.87*9.8, 0,0,0])
    quad.motorSpeedsFromInputs(u)
    quad.startThread()

    for i in range(10):
        time.sleep(1)
        print(quad._state[2]) # altitude
        
    quad.stopThread()
        

    