#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import random
import time
from feature_env import Env
from motion import MotionModels
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class UKF:
    def __init__(self, motion, dt=1.0):
        self.motion = motion
        self.alpha = motion.get_alpha()
        self.dt = dt
        self.qt = np.diag([
            10.,
            np.deg2rad(1.0),  # variance of yaw angle
            1
        ]) ** 2
        # self.qt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def localization_with_known_correspondences(self, mut_1, sigmat_1, ut, zt, m):
        """
        The unscented Kalman filter (EKF) localization algorithm, formulated here
        for a feature-based map and a robot equipped with sensors for measuring range and
        bearing. This version assumes knowledge of the exact correspondences (p. 221, Probabilistic Robotics)
        The _t denotes a time step t, its absence means the time step t-1.
        :param mut_1: pose at previous step
        :param sigmat_1: variance at previous step
        :param ut: taken action
        :param zt: got observation
        :param m: map (knowledge about coordinates of landmarks)
        :return:
        """
        dt = self.dt
        vt, wt = ut
        theta = mut_1[2]
        Gt = self.motion.get_jacobian(mut_1, ut)
        if wt != 0:
            Vt = np.array([
                [
                    (-np.sin(theta) + np.sin(theta+wt*dt)) / wt,
                    (np.sin(theta) - np.sin(theta+wt*dt)) * vt / wt ** 2 + np.cos(theta + wt * dt) * vt * dt / wt
                ],
                [
                    (np.cos(theta) - np.cos(theta + wt * dt)) / wt,
                    - (np.cos(theta) - np.cos(theta + wt * dt)) * vt / wt ** 2 + np.sin(theta + wt * dt) * vt * dt / wt
                ],
                [0, dt]
            ])
        else:
            Vt = np.array([
                [
                    np.cos(theta),
                    0
                ],
                [
                    0,
                    np.sin(theta)
                ],
                [0, 0]
            ])
        Mt = np.array([
            [self.alpha[0] * vt**2 + self.alpha[1] * wt ** 2, 0],
            [0, self.alpha[2] * vt**2 + self.alpha[3] * wt ** 2]
        ])
        if wt != 0:
            mut_hat = mut_1 + np.array([
                -vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * dt),
                vt / wt * np.cos(theta) - vt / wt * np.cos(theta + wt * dt),
                wt * dt
            ])
        else:
            mut_hat = mut_1 + np.array([
                vt * np.cos(theta) * dt,
                vt * np.sin(theta) * dt,
                0
            ])
        sigmat_hat = Gt @ sigmat_1 @ Gt.T + Vt @ Mt @ Vt.T
        Qt = self.qt
        z_array = []
        S_array = []
        for i, zti in enumerate(zt):
            j = ct[i]
            mjx = m[j][0]
            mjy = m[j][1]
            q = (mjx - mut_hat[0]) ** 2 + (mjy - mut_hat[1]) ** 2
            zti_hat = np.array(
                [
                    q ** 0.5,
                    np.arctan2(mjy - mut_hat[1], mjx - mut_hat[0]) - mut_hat[2],  # -theta
                    i
                ]
            ).T
            Hti = np.array([
                [-(mjx - mut_hat[0])/q ** 0.5, -(mjy - mut_hat[1])/q ** 0.5, 0],
                [(mjy - mut_hat[1])/q, -(mjx - mut_hat[0])/q, -1],
                [0, 0, 0],
            ])
            Sti = Hti @ sigmat_hat @ Hti.T + Qt
            Kti = sigmat_hat @ Hti.T @ np.linalg.inv(Sti)
            mut_hat = mut_hat + Kti @ (zti - zti_hat)
            sigmat_hat = (np.eye(3) - Kti @ Hti) @ sigmat_hat
            z_array.append(zti_hat)
            S_array.append(Sti)
        mut = mut_hat
        sigmat = sigmat_hat
        pzt = 1
        for i in range(len(zt)):
            pzt *= np.linalg.det(2 * np.pi * S_array[i]) ** 0.5 * np.exp(-0.5 * (zt[i] - z_array[i]).T @ S_array[i] @ (zt[i] - z_array[i]))
        return mut, sigmat, pzt

    def get_qt(self):
        return self.qt

    def set_qt(self, qt):
        self.qt = qt


def commands():
    """
    :return: list of commands
    """
    u = []
    for i in range(600):
        u.append([5, 0.05])
    return u


def main():
    landmarks_num = 1
    try:
        landmarks_num = int(sys.argv[1])
    except Exception:
        print("You did not entered number of landmarks. Default: 1.")

    dt = 0.1
    # initialization
    initial_pose = [400, 200, 0]
    # initial pose estimate
    mu = initial_pose
    # initial real pose
    x = initial_pose
    # initial variance
    sigma = np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ])
    # motion model
    motion = MotionModels(dt, distribution="normal")
    # localization algorithm
    ukf = UKF(motion, dt)
    # observation noise
    qt = ukf.get_qt()
    # environment
    env = Env(*initial_pose, landmarks_num=landmarks_num, qt=qt)
    # command list
    cmds = commands()
    # map
    m = env.get_landmarks()
    # iterate over command list
    for i, u in enumerate(cmds):
        # Move and observe.
        x = motion.sample_motion_model_velocity(u, x)
        env.set_pose(x)
        z = env.get_observations()[:2]  # no need of correspondences
        # Estimate pose.
        mu, sigma, p_t = ukf.localization_with_known_correspondences(mu, sigma, u, z, m)
        # Visualize
        env.draw_step(i, len(cmds), mu, sigma)


if __name__ == "__main__":
    main()
