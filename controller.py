"""
Author: Mohamed Abdelkader
Email: mohamedashraf123@gmail.com
Date: Sept, 2022

Description:
    Implementation of cascade PID controllers for position control of quadrotor

"""

from spatialmath import SO3, SE3
import numpy as np
import time
import datetime
import threading

class PID:
    def __init__(self, kp, ki, kd, maxInt = 100):
        # Gains
        self._kp = kp
        self._ki = ki
        self._kd = kd

        # Integral term
        self._int = 0
        # Maximum PID output (to avoid integral wind up)
        self._maxInt = maxInt

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
        self._x_pid = PID(params['xy_kp'], 0,0, maxInt=50)
        self._y_pid = PID(params['xy_kp'], 0,0, maxInt=50)
        self._vx_pid = PID(params['vxy_pid'][0], params['vxy_pid'][1], params['vxy_pid'][2], maxInt=50)
        self._vy_pid = PID(params['vxy_pid'][0], params['vxy_pid'][1], params['vxy_pid'][2], maxInt=50)
        # Z PID
        self._z_pid = PID(params['z_kp'], 0,0, maxInt=50)
        self._vz_pid = PID(params['vz_pid'][0], params['vz_pid'][1], params['vz_pid'][2], maxInt=50)

        # roll/pitch/yaw PID
        self._roll_pid = PID(params['roll_kp'], 0, 0, maxInt=200)
        self._pitch_pid = PID(params['pitch_kp'], 0, 0, maxInt=200)
        self._yaw_pid = PID(params['yaw_kp'],0,0, maxInt=200)

        # Roll/pitch/yaw rate PID
        self._roll_rate_pid = PID(params['roll_speed_pid'][0], params['roll_speed_pid'][1], params['roll_speed_pid'][2], maxInt=50)
        self._pitch_rate_pid = PID(params['pitch_speed_pid'][0], params['pitch_speed_pid'][1], params['pitch_speed_pid'][2], maxInt=50)
        self._yaw_rate_pid = PID(params['yaw_speed_pid'][0], params['yaw_speed_pid'][1], params['yaw_speed_pid'][2], maxInt=50)

        # Maximum XY velocity, m/s
        self._maxXYVel = 5.0
        # Maximum Z velocity, m/s
        self._maxZVel = 2.0

        # Maximum tilt angles, rad
        self._maxTilt = 45*np.pi/180.
        # Maximum angular rate, rad/s
        self._maxTiltRate = 200*np.pi/180.
        self._maxYawRate = 100*np.pi/180.

        # Quadcopter state (feedback)
        # x= [x y z x_dot y_dot z_dot phi theta psi p q r]
        self._state = np.zeros(12)

        # Moments scaler
        self._M_scaler = 1.0

        # Desired position
        self._des_pos = np.zeros(3)
        # desired yaw angle, rad
        self._des_yaw = 0

        self._time = datetime.datetime.now()

        self._thread_object = None

        self._run = True


    def initPIDs(self):
        pass

    def updateState(self, state):
        self._state = state

    def setDesiredPos(self, des_pos):
        self._des_pos = des_pos

    def setDesiredYaw(self, des_yaw):
        self._des_yaw = des_yaw

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
        print("acc: ", acc)
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

    def wrapAngle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def update(self, dt):
        """
        Update all PID controllers

        Params:
            des_pos: 3x1 numpy.ndarray desired 3D position
            des_yaw: (float) desired yaw angle, rad
            state: 12x1 numpy.ndarray state feedback
            dt: time step, seconds
        """
        # print("des_pos: ", self._des_pos)
        e_px = self._des_pos[0] - self._state[0]
        e_py = self._des_pos[1] - self._state[1]
        e_pz = self._des_pos[2] - self._state[2]
        # print("position errors: ", e_px, e_py, e_pz)

        vx_sp = np.clip(self._x_pid.update(e_px,dt), -self._maxXYVel, self._maxXYVel)
        vy_sp = np.clip(self._y_pid.update(e_py,dt), -self._maxXYVel, self._maxXYVel)
        vz_sp = np.clip(self._z_pid.update(e_pz,dt), -self._maxZVel,self._maxZVel)
        # print("vel sp: ", vx_sp, vy_sp, vz_sp)

        e_vx = vx_sp - self._state[3]
        e_vy = vy_sp - self._state[4]
        e_vz = vz_sp - self._state[5]

        ax_sp = self._vx_pid.update(e_vx,dt)
        ay_sp = self._vy_pid.update(e_vy,dt)
        az_sp = self._vz_pid.update(e_vz,dt)

        des_acc = np.array([ax_sp, ay_sp, az_sp])
        _, rpy, thrust = self.acc2RotAndThrust(des_acc, self._des_yaw)

        roll_des = self.wrapAngle(rpy[0])
        roll_des = np.clip(roll_des, -self._maxTilt, self._maxTilt)
        pitch_des = self.wrapAngle(rpy[1])
        pitch_des = np.clip(pitch_des, -self._maxTilt, self._maxTilt)
        yaw_des = rpy[2]
        yaw_des = self.wrapAngle(yaw_des)

        roll_rate_sp = np.clip(self._roll_pid.update(roll_des-self._state[6], dt), - self._maxTiltRate, self._maxTiltRate)
        pitch_rate_sp = np.clip(self._pitch_pid.update(pitch_des-self._state[7], dt), -self._maxTiltRate, self._maxTiltRate)
        yaw_rate_sp = np.clip(self._yaw_pid.update(yaw_des-self._state[8], dt), -self._maxYawRate, self._maxYawRate)

        # Body moments
        mx = self._M_scaler*self._roll_rate_pid.update(roll_rate_sp - self._state[9], dt)
        my = self._M_scaler*self._pitch_rate_pid.update(pitch_rate_sp - self._state[10], dt)
        mz = self._M_scaler*self._yaw_rate_pid.update(yaw_rate_sp - self._state[11], dt)

        u = np.array([thrust, mx, my, mz])
        # print("u:", u)

        return u
    
    def threadRun(self,update_rate):
        rate = update_rate
        last_update = self._time
        while(self._run==True):
            time.sleep(0)
            self._time = datetime.datetime.now()
            if (self._time - last_update).total_seconds() > rate:
                self.update(update_rate)
                last_update = self._time

    def startThread(self,update_rate=0.005):
        self._thread_object = threading.Thread(target=self.threadRun,args=(update_rate,))
        self._thread_object.start()

    def stopThread(self):
        self._run = False

if __name__ == "__main__":
    params={'xy_kp': 1.0, 'z_kp': 1.0,
    'vxy_pid':[1.0,0.2, 0.004], 'vz_pid': [1.0, .02, 0.003],
    'roll_kp': 6.0, 'pitch_kp': 6.0, 'yaw_kp': 2.0,
    'roll_speed_pid': [0.15, 0.2, 0.003],
    'pitch_speed_pid': [0.15, 0.2, 0.003], 
    'yaw_speed_pid': [0.15, 0.2, 0.0]}
    cont = Controller(params)
    cont.updateState(np.zeros(12))
    cont.setDesiredPos(np.array([0,0,1]))
    cont.setDesiredYaw(45*np.pi/180)
    cont.startThread()

    for i in range(10):
        time.sleep(1)
        cont.setDesiredPos(np.array([0,1,0+i+1]))

    cont.stopThread()
        