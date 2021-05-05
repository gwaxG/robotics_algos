#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import random
import time
from scipy.linalg import cholesky
from feature_env import Env
from motion import MotionModels
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class SigmaPoints:
    def __init__(self, n, alpha=0.1, beta=2., kappa=1.):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.sqrt = cholesky
        self.subtract = np.subtract
        self._compute_weights()

    def sigma_points(self, x, P):

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k+1] = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas

    def _compute_weights(self):
        n = self.n
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        c = .5 / (n + lambda_)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)


class UKF:
    def __init__(self, motion, dt=1.0):
        L = 7  # state, control and observation dimensions
        self.sigmas = SigmaPoints(L)
        self.motion = motion
        self.alpha = motion.get_alpha()
        self.dt = dt
        self.qt = np.diag([
            10.,
            np.deg2rad(1.0)
        ]) ** 2

    def get_sigma(self, sigma_t, m_t, q_t):
        z1 = np.zeros((len(sigma_t), len(sigma_t)), dtype=float)
        z2 = np.zeros((len(m_t), len(m_t)), dtype=float)
        z3 = np.zeros((len(q_t), len(q_t)), dtype=float)
        return np.asarray(np.bmat([[sigma_t, z2, z3], [z1, m_t, z3], [z1, z2, q_t]]))

    def localization(self, mut_1, sigmat_1, ut, zt, m):
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
        Qt = self.qt
        mu_a_t_1 = np.array(
            [
                mut_1.T,
                np.array([0, 0]).T,
                np.array([0, 0]).T
            ]
        )
        sigma_a_t_1 = self.get_sigma()

        sigma_points = self.sigmas.sigma_points(mu_a_t_1, sigma_a_t_1)
        
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
        z = env.get_observations(without_correspondences=True)
        # Estimate pose
        mu, sigma, p_t = ukf.localization(mu, sigma, u, z, m)
        # Visualize
        env.draw_step(i, len(cmds), mu, sigma)


if __name__ == "__main__":
    main()
