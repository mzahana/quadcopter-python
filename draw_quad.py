"""
Author: Mohamed Abdelkader, mohamedashraf123@gmail.com
Copyright 2022

Description:
Implementation of quadrotor visualization class
"""

import numpy as np
from spatialmath import SE3, SO3
# import matplotlib.pyplot as plt
# plt.style.use('seaborn')

class QuadPlot:
    """
    Draws a quadcopter, given a position and orientation matrix R
    """
    def __init__(self, ax, quad, color='r'):
        """
        Initialize quadcopter plot

        Params:
            ax: plotting axis
            color: [str] quadcopter color
        """
        self._ax = ax # plotting axis
        self._quad = quad # Quadcopter object
        self._prop_l = quad._prop_l # Propeller diameter
        self._arm_l = quad._L*2 # Arm length in meter (from motor to motor)  
        t = quad.getPosition()
        rot = quad.getR()
        R = SO3(rot)
        self._T = SE3.Rt(R,t) # Homogenous transformation matrix of the quadrotor

        # Drone body axes
        self._xb = np.array([1.0, 0.0, 0.0])
        self._yb = np.array([0.0, 1.0, 0.0])
        self._zb = np.array([0.0, 0.0, 1.0])

        # End points of 1st arm, along the body x-axis (that hold motor1 and motor2)
        self._armX_start= self._arm_l/2*self._xb
        self._armX_end= -self._arm_l/2*self._xb

        # 2nd arm (that hold motor3 and motor4)
        self._armY_start= self._arm_l/2*self._yb
        self._armY_end= -self._arm_l/2*self._zb

        self._m1_w = []
        self._m2_w = []
        self._m3_w = []
        self._m4_w = []

        self._color=color


    def drawArms(self):
        # Body frame arm coordinates
        # End points of 1st arm (that holds motor1 and motor2)
        a1_x0= self._arm_l/2*self._xb # along +ve body x-axis
        a1_x1= -self._arm_l/2*self._xb # along -ve body x-axis
        a1_b = np.array([a1_x0, a1_x1]).T # put them in one matrix

        # 2nd arm (that holds motor3 and motor4)
        a2_x0= self._arm_l/2*self._yb # along +ve body y-axis
        a2_x1= -self._arm_l/2*self._yb # along -ve body y-axis
        a2_b = np.array([a2_x0, a2_x1]).T

        # Transform points from body frame to the world frame
        a1_w = (self._T*a1_b).squeeze()
        a2_w = (self._T*a2_b).squeeze()

        # Plot
        self._ax.plot([a1_w[0][0], a1_w[0][1]],
                    [a1_w[1][0], a1_w[1][1]],
                    [a1_w[2][0], a1_w[2][1]], color='r')

        self._ax.plot([a2_w[0][0], a2_w[0][1]],
                    [a2_w[1][0], a2_w[1][1]],
                    [a2_w[2][0], a2_w[2][1]], color='k')

    def drawPropellers(self):
        # Consider a plus-shape quadcopter

        theta = np.linspace(0,2*np.pi,200)
        l = len(theta)
        # Coordinates in body frame
        # Front motor
        m1 = np.array([self._arm_l/2 + self._prop_l/2*np.cos(theta), # x-coordinates
                        0 + self._prop_l/2*np.sin(theta),               # y coordinates
                        np.zeros(l)])                                      # z ccordinates

        # back motor
        m2 = np.array([-self._arm_l/2 + self._prop_l/2*np.cos(theta), # x-coordinates
                        0 + self._prop_l/2*np.sin(theta),               # y coordinates
                        np.zeros(l)])                                      # z ccordinates


        # Left motor
        m3 = np.array([0 + self._prop_l/2*np.cos(theta),                    # x-coordinates
                        self._arm_l/2 + self._prop_l/2*np.sin(theta),      # y coordinates
                        np.zeros(l)])                                      # z ccordinates

        # Right motor
        m4 = np.array([0 + self._prop_l/2*np.cos(theta),                    # x-coordinates
                        -self._arm_l/2 + self._prop_l/2*np.sin(theta),      # y coordinates
                        np.zeros(l)])                                      # z ccordinates

        # Coordinates in world frame
        self._m1_w = self._T*m1
        self._m2_w = self._T*m2
        self._m3_w = self._T*m3
        self._m4_w = self._T*m4

        self._ax.plot(self._m1_w[0,:],
                        self._m1_w[1,:],
                        self._m1_w[2,:], color='r')
        self._ax.plot(self._m2_w[0,:],
                        self._m2_w[1,:],
                        self._m2_w[2,:], color='k')

        self._ax.plot(self._m3_w[0,:],
                        self._m3_w[1,:],
                        self._m3_w[2,:], color='k')

        self._ax.plot(self._m4_w[0,:],
                        self._m4_w[1,:],
                        self._m4_w[2,:], color='k')

    def update(self):
        R = SO3(self._quad.getR())
        t = self._quad.getPosition()
        self._T = SE3.Rt(R,t)

        self.drawArms()
        self.drawPropellers()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from quadcopter import Quadcopter
    q = Quadcopter()
    q.setPosition(np.array([1,2,3]))
    q.setRPY(np.array([45.*np.pi/180,0,0]))

    # plt.style.use('seaborn')
    fig = plt.figure('Quad')
    ax = fig.add_subplot(111, projection='3d')

    quad = QuadPlot(ax,q)
    quad.update()
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()