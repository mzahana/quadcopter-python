"""
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept 2022

Description:
    Implements a class that stores data of a quadcopter including:
    physical characterstics, states, and control data
"""
import numpy as np
from spatialmath import SO3

class Quadcopter:
    def __init__(self, name="quad1", L=0.22, m=2.0, kf=1.8e-5, km=1.e-6, r=0.1, max_rpm=13000, prop_l=0.2286) -> None:
        """
        L: (float) arm length, half of the body length (assuming symmetric quadcopter), meter
        kf: (float) Motor force constant
        km: (float) Motor moment constant
        m: (float) Mass, Kg
        r: (float) Radius of sphere that is used to estimate inertia
        max_m_rpm: (int) Maximum motor RPM
        prop_l: (float) Propeller diameter in meters
        """
        self._name=name
        self._L=L
        self._m = m
        self._kf= kf
        self._km= km
        self._gamma = self._km/self._kf
        self._max_rpm = max_rpm
        self._max_rps = self._max_rpm * 0.10472 # in rad/sec
        self._r = r
        self._prop_l = prop_l

        # Inertia is estimated according to the following reference
        # Quadrotor Dynamics and Control Rev , Randal Beard
        # https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=2324&context=facpub
        ixx = (2*self._m*self._r**2/5) + (2*self._L**2/self._m)
        iyy = ixx
        izz = (2*self._m*self._r**2/5) + (4*self._L**2/self._m)
        self._I = np.array([[ixx,0,0],[0,iyy,0],[0,0,izz]])
        self._Iinv = np.linalg.inv(self._I)

        # Current state
        # x= [x y z x_dot y_dot z_dot phi theta psi phi_dot theta_dot psi_dot]
        self._state = np.zeros(12)

        # Rotation matrix
        self._R = np.eye(3,3)

        # Inputs
        # u_1 = total body thrust, (float), Newton
        self._u1 = 0
        # Moments u2=[ux,uy,uz]
        self._u2 = np.array([0,0,0])

        # Motors speeds in rad/sec
        self._motor_speeds = np.zeros(4)

        # Control allocation matrix
        # Transforms motor forces to thrust and moments
        self._C = np.array([[1,1,1,1],
                            [0,0,self._L, -self._L],
                            [-self._L, self._L, 0,0],
                            [-self._gamma, -self._gamma,self._gamma,self._gamma]])
        # Transforms thrust and moments to motor forces
        self._Cinv = np.linalg.inv(self._C)

    def inputsFromMotorSpeeds(self) -> np.ndarray:
        """
        Calculate inputs self._u1 from motors speeds

        Params:
        w: (4x1 np.ndarray) motors speeds
        """
        u = self._C.dot(self._kf*self._motor_speeds**2)
        self._u1 = u[0]
        self._u2 = u[1:]
        return u

    def setMotorSpeedsFromInputs(self,u) -> np.ndarray:
        """
        Calculate motor speeds from inputs u=[self._u1, self._u2]

        Params:
            u: (4x1 np.ndarray) u=[u1(1x1)=thrust, u2(3x1)=moments]
        """
        F=self._Cinv.dot(u)
        F = np.clip(F,0,np.inf)
        # print("F: ", F)
        self._motor_speeds = np.sqrt(F/self._kf)
        self._motor_speeds=np.clip(self._motor_speeds, 0, self._max_rps)

    def getState(self) -> np.ndarray:
        return self._state
    
    def getPosition(self) -> np.ndarray:
        return self._state[0:3]
    
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
        self._R = R.R
        return self._R

    def getRPY(self) -> np.ndarray:
        """
        Returns the current [roll,pitch,yaw] angles from the state vector
        """
        return self._state[6:9]

    def getAngularVel(self) -> np.ndarray:
        """
        Returns the current angular velocity from the state vector
        """
        return self._state[9:12]

    def getLinearVel(self) -> np.ndarray:
        """
        Returns linear velocity
        """
        return self._state[3:6]

    def getTotalThrust(self) -> float:
        return self._u1

    def getMoments(self) -> np.ndarray:
        return self._u2

    def mass(self):
        return self._m

    def setState(self, state):
        self._state = state
        self.getR() # Update rotation matrix

    def setRPY(self, rpy):
        """
        Update euler angles in the state vector, and the roation matrix self._R accordingly
        """
        self._state[6] = rpy[0] # roll
        self._state[7] = rpy[1] # pitch
        self._state[8] = rpy[2] # yaw

        # Update rotation matrix as well
        R = SO3.Rz(rpy[2])*SO3.Ry(rpy[1])*SO3.Rx(rpy[0])
        self._R = R.R

    def setPosition(self, p):
        """
        Update position in the state vector
        """
        self._state[0] = p[0]
        self._state[1] = p[1]
        self._state[2] = p[2]
