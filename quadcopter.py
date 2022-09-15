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
from spatialmath import SE3
import matplotlib.pyplot as plt

class Quadcopter:
    def __init__(self) -> None:
        self._prop_diam = 0.228 # Properller diameter in meter
        self._frame_len = 0.45  # Frame length, motor-to-motor, in meter

    def drawPropellers(self, pos, rot):
        """Draws all four propeller as 4 circles

        Params
        --
        pos: 3D position
        rot: 3D rotation matrix
        """
        
        # Define circles points
        theta = np.linspace(0, 2 * np.pi, 201)
        prop1_x = self._frame_len/2. + self._frame_len/2.*np.cos(theta)
        prop1_y = 0 + self._frame_len/2.*np.sin(theta)
        prop1_z=0