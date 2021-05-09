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


class Support:
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
        print(sigmas)
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
        self.sigmas = Support(L)
        self.motion = motion
        self.alpha = motion.get_alpha()
        self.dt = dt
        self.qt = np.diag([
            10.,
            np.deg2rad(1.0)
        ]) ** 2

    def get_sigma(self, sigma_t, m_t, q_t):
        l1 = len(sigma_t)
        l2 = len(m_t)
        l3 = len(q_t)
        l = l1 + l2 + l3
        z = np.zeros((l1, l - l1), dtype=float)
        z21 = np.zeros((l2, l1), dtype=float)
        z22 = np.zeros((l2, l3), dtype=float)
        return np.asarray(np.bmat([[sigma_t, z], [z21, m_t, z22], [z21, z22, q_t]]))

    def pass_points(self, points, ut):
        passages  = []
        for point in points:
            x = point[:3]
            u = point[3:5]
            passages.append(self.motion.sample_motion_model_velocity(x, u))
        return np.array(passages)

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
        # data preparation
        dt = self.dt
        vt, wt = ut
        theta = mut_1[2]
        # line 2
        Mt = np.array([
            [self.alpha[0] * vt**2 + self.alpha[1] * wt ** 2, 0],
            [0, self.alpha[2] * vt**2 + self.alpha[3] * wt ** 2]
        ])
        # line 3
        Qt = self.qt
        # line 4
        mu_a_t_1 = np.array([mut_1.tolist() + ut.tolist() + zt.tolist()])
        # line 5
        sigma_a_t_1 = self.get_sigma(sigmat_1, Mt, Qt)

        # Generate sigma points
        # line 6
        sigma_points = self.sigmas.sigma_points(mu_a_t_1, sigma_a_t_1)
        
        # Pass sigma points through motion model and compute Gaussian statistics
        # line 7
        khi_x_t_hat = self.pass_points(sigma_points, ut)
        exit()

        # predict step
        # Predict observations at sigma points and compute Gaussian statistics
        # line 10

        # update step
        # Update mean and covariance

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

def test():
    dt = 0.1
    motion = MotionModels(dt, distribution="normal")
    ukf = UKF(motion, dt)
    mu = np.array([5., 1., 0.])
    sigma = np.array([
        [10, 0, 0], 
        [0, 10, 0],
        [0, 0, 10],
    ])
    u = np.array([1.0, 0.1])
    z = np.array([4.0, np.pi/2])
    m = np.array([[5.0, 5.0]])
    mu, sigma, p_t = ukf.localization(mu, sigma, u, z, m)
    print(mu, sigma)

if __name__ == "__main__":
    # main()
    test()
