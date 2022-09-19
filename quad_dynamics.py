"""
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
Implementation of quadrotor dynamics

"""

from cmath import isnan
import numpy as np
from spatialmath import SE3, SO3
import matplotlib.pyplot as plt
import scipy.integrate
import time
import datetime
import threading

class Dynamics:
    def __init__(self, quad) -> None :
        """
        Params:
            quad: Quadcopter object from quadcopter.py
        """
        self._quad=quad

        self._g = 9.18 # gravity m/s/s

        
        self._ode =  scipy.integrate.ode(self.stateDot).set_integrator('vode',nsteps=500,method='bdf')

        self._time = datetime.datetime.now()

        self._thread_object = None

        self._run = True


    def wrapAngle(self,val):
        if isnan(val):
            return 0
        # print("ang: ", val)
        if val < -np.pi or val > np.pi:
            return( ( val + np.pi) % (2 * np.pi ) - np.pi )
        return val

    
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


    def stateDot(self, time, state):
        #            [0 1 2  3     4      5    6    7    8  9 10 11]
        # States, x= [x y z x_dot y_dot z_dot phi theta psi p q r]
        x_dot = np.zeros(12)

        # Linear velocities
        v = self._quad.getLinearVel()
        x_dot[0] = v[0]
        x_dot[1] = v[1]
        x_dot[2] = v[2]
        # Linear accelerations
        self._quad.inputsFromMotorSpeeds()
        u1=self._quad.getTotalThrust()
        # print("u1: ", u1)
        R = self._quad.getR()
        m = self._quad.mass()
        acc = np.array([0,0, -self._g]) + (1/m) * R.dot(np.array([0,0,u1]))
        x_dot[3] = acc[0]
        x_dot[4] = acc[1]
        x_dot[5] = acc[2]
        # angular rates
        rpy = self._quad.getRPY()
        omega = self._quad.getAngularVel()
        rpy_dot = self.calcRPYdot(rpy, omega)
        x_dot[6] = rpy_dot[0]
        x_dot[7] = rpy_dot[1]
        x_dot[8] = rpy_dot[2]
        # Angular acceleration
        u2 = self._quad.getMoments()
        I = self._quad._I
        Iinv = self._quad._Iinv
        omega_dot = np.dot(Iinv, ( u2- np.cross( omega, np.dot(I, omega) ) ) )
        x_dot[9] = omega_dot[0]
        x_dot[10] = omega_dot[1]
        x_dot[11] = omega_dot[2]

        return x_dot

    def update(self, dt):
        self._ode.set_initial_value(self._quad.getState(),0)
        state = self._ode.integrate(self._ode.t + dt)
        state[6] = self.wrapAngle(state[6])
        state[7] = self.wrapAngle(state[7])
        state[8] = self.wrapAngle(state[8])
        state[2] = max(0,state[2]) # to avoid going below ground level (z=0)
        self._quad.setState(state)

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
    from quadcopter import Quadcopter
    q = Quadcopter()
    dyn = Dynamics(q)
    init_state = np.zeros(12)
    q.setState(init_state)
    u = np.array([1.9*9.8, 0,0,0])
    q.setMotorSpeedsFromInputs(u)
    q.inputsFromMotorSpeeds()
    dyn.startThread()

    for i in range(10):
        time.sleep(1)
        xyz=q.getPosition()
        print(xyz[2]) # altitude
        
    dyn.stopThread()
        

    