"""
Course code: CS 489
Course title: Introduction to Unmanned Aerial Systems
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
    Implementation of cascade PID controllers for position control of quadrotor

"""

from spatialmath import SO3, SE3
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
    def __init__(self, params):
        # params={'xy_kp', 'z_kp', 'vxy_pid', 'vz_pid', 'roll_kp', 'pitch_kp' 'yaw_kp', 'roll_speed_pid', 'pitch_speed_pid' 'yaw_speed_pid'}
        # XY PIDs
        self._x_pid = PID(params['xy_kp'], 0,0)
        self._y_pid = PID(params['xy_kp'], 0,0)
        self._vx_pid = PID(params['vxy_pid'][0], params['vxy_pid'][1], params['vxy_pid'][2])
        self._vy_pid = PID(params['vxy_pid'][0], params['vxy_pid'][1], params['vxy_pid'][2])
        # Z PID
        self._z_pid = PID(params['z_kp'], 0,0)
        self._vz_pid = PID(params['vz_pid'][0], params['vz_pid'][1], params['vz_pid'][2])

        # roll/pitch/yaw PID
        self._roll_pid = PID(params['rp_kp'], 0, 0)
        self._pitch_pid = PID(params['rp_kp'], 0, 0)
        self._yaw_pid = PID(params['yaw_kp'],0,0)

        # Roll/pitch/yaw rate PID
        self._roll_rate_pid = PID(params['roll_speed_pid'][0], params['roll_speed_pid'][1], params['roll_speed_pid'][2])
        self._pitch_rate_pid = PID(params['pitch_speed_pid'][0], params['pitch_speed_pid'][1], params['pitch_speed_pid'][2])
        self._yaw_rate_pid = PID(params['yaw_speed_pid'][0], params['yaw_speed_pid'][1], params['yaw_speed_pid'][2])

        # Maximum XY velocity, m/s
        self._maxXYVel = 5.0
        # Maximum Z velocity, m/s
        self._maxZVel = 2.0

        # Maximum tilt angles, rad
        self._maxTilt = 45*np.pi/180.
        # Maximum angular rate, rad/s
        self._maxAngRate = 200*np.pi/180.

        # Quadcopter state (feedback)
        # x= [x y z x_dot y_dot z_dot phi theta psi p q r]
        self._state = np.zeros(12)

        # Moments scaler
        self._M_scaler = 1.0


    def initPIDs(self):
        pass

    def setMomentScaler(self, val):
        self._M_scaler = val

    def acc2RotAndThrust(self, acc, yaw):
        """
        Converts desired accelration vector & yaw to full rotation and thrust(in body frame)
        Params:
            acc: 3x1 acceleration in inertial frame m/s/s
            yaw: (float) desired yaw in rad
        Returns:
            rotMat: 3x3 Desired rotation matrix
            rpy: 3x1 Desired [roll, pitch, yaw] in ZYX order, radian
            thrust: (float) Total desired thrust in body frame, Newton

        """
        proj_xb_des = np.array([np.cos(yaw), np.sin(yaw), 0])
        zb_des = acc / np.linalg.norm(acc)
        v = np.cross(zb_des, proj_xb_des)
        yb_des = v/np.linalg.norm(v)
        v = np.cross(yb_des,zb_des)
        xb_des = v/np.linalg.norm(v)

        rotMat = np.array([[xb_des[0], yb_des[0], zb_des[0]],
                        [xb_des[1], yb_des[1], zb_des[1]],
                        [xb_des[2], yb_des[2], zb_des[2]]])

        R = SO3(rotMat)
        rpy = R.rpy(order='zyx')
        thrust = np.dot(acc, zb_des)

        return (rotMat, rpy, thrust)

    def updateState(self, state):
        self._state = state

    def update(self, des_pos, des_yaw, state, dt):
        """
        Update all PID controllers

        Params:
            des_pos: 3x1 numpy.ndarray desired 3D position
            des_yaw: (float) desired yaw angle, rad
            state: 12x1 numpy.ndarray state feedback
            dt: time step, seconds
        """
        e_px = des_pos[0] - state[0]
        e_py = des_pos[1] - state[1]
        e_pz = des_pos[2] - state[2]

        vx_sp = self._x_pid.update(e_px,dt)
        vy_sp = self._y_pid.update(e_py,dt)
        vz_sp = self._z_pid.update(e_pz,dt)

        e_vx = vx_sp - state[3]
        e_vy = vy_sp - state[4]
        e_vz = vz_sp - state[5]

        ax_sp = self._vx_pid.update(e_vx,dt)
        ay_sp = self._vy_pid.update(e_vy,dt)
        az_sp = self._vz_pid.update(e_vz,dt)

        des_acc = np.array([ax_sp, ay_sp, az_sp])
        _, rpy, thrust = self.acc2RotAndThrust(des_acc, des_yaw)

        roll_des = rpy[0]
        pitch_des = rpy[1]
        yaw_des = rpy[2]

        roll_rate_sp = self._roll_pid.update(roll_des-state[6], dt)
        pitch_rate_sp = self._pitch_pid.update(pitch_des-state[7], dt)
        yaw_rate_sp = self._yaw_pid.update(yaw_des-state[8], dt)

        # Body moments
        mx = self._M_scaler*self._roll_rate_pid.update(roll_rate_sp - state[9], dt)
        my = self._M_scaler*self._pitch_rate_pid.update(pitch_rate_sp - state[10], dt)
        mz = self._M_scaler*self._yaw_rate_pid.update(yaw_rate_sp - state[11], dt)

        u = np.array([thrust, mx, my, mz])

        return u
        