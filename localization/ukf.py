#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import random
import time
from scipy.linalg import cholesky
from environment import Env
from motion import MotionModels
import numpy as np
from scipy.spatial.transform import Rotation as Rot

"""
Single feature UKF localization.
Page 221 of Probabilistic Robotics.
"""


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
            sigmas[k+1] = self.subtract(x[0], -U[k])
            sigmas[n+k+1] = self.subtract(x[0], U[k])
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
        self.qt = None
        self.get_observations = None

    def set_obs(self, f):
        self.get_observations = f

    def _get_sigma(self, sigma_t, m_t, q_t):
        l1 = len(sigma_t)
        l2 = len(m_t)
        l3 = len(q_t)
        l = l1 + l2 + l3
        z = np.zeros((l1, l - l1), dtype=float)
        z21 = np.zeros((l2, l1), dtype=float)
        z22 = np.zeros((l2, l3), dtype=float)
        return np.asarray(np.bmat([[sigma_t, z], [z21, m_t, z22], [z21, z22, q_t]]))

    def get_sigma(self, sigma_t, m_t, q_t):
        l1 = len(sigma_t)
        l2 = len(m_t)
        l3 = len(q_t)
        l = l1 + l2 + l3
        z = []
        for i in range(l):
            line = []
            for j in range(l):
                if j < l1:
                    if i < l1:
                        line.append(sigma_t[i][j])
                    else:
                        line.append(0)
                if l1 <= j < l1 + l2:
                    if l1 <= i < l1 + l2:
                        line.append(m_t[i-l1][j-l1])
                    else:
                        line.append(0)
                if l1 + l2 <= j < l1 + l2 + l3:
                    if l1 + l2 <= i < l1 + l2 + l3:
                        line.append(q_t[i - l1- l2][j - l1 - l2])
                    else:
                        line.append(0)
            z.append(line)
        return np.array(z)

    def pass_points(self, points, ut):
        passages  = []
        for point in points:
            x = point[:3]
            u = point[3:5]
            passages.append(self.motion.sample_motion_model_velocity(x, u))
        return np.array(passages)


    def estimate_mean(self, points):
        mu = np.zeros(3)
        for i, p in enumerate(points):
            mu += p * self.sigmas.Wm[i]
        return mu     

    def estimate_sigma(self, khi, mu):
        sigma = np.zeros((3,3))
        for i in range(len(khi)):
            sigma += self.sigmas.Wc[i] * (khi[i] - mu) * (khi[i] - mu).T
        return sigma

    def pass_observations(self, poses, sigma_points):
        z = []
        for i in range(len(sigma_points)):
            # last two points
            p = poses[i]
            zt = sigma_points[i][5:]
            z = self.get_observations(p, False, True) + zt
        return np.array(z)

    def estimate_obs(self, z_t_av):
        z = np.zeros(len(z_t_av))
        for i in range(len(z_t_av)):
            z += z_t_av[i] * self.sigmas.Wm[i]
        return z

    def estimate_gain_sigma(self, khi, mut, z_av, z_hat):
        s = np.zeros(np.matmul(z_av, z_av.T).shape)
        for i in range(len(z_hat)):
            inter = (z_av[i] - z_hat).tolist()
            inter.append(0.)
            inter = np.array(inter)
            inter = np.array(inter)
            s += self.sigmas.Wc[i] * np.matmul(khi[i] - mut, inter.T)  # modification, no .T
        return s

    def estimate_uncertainty_ellipse(self, z_av, z_hat):
        st = np.zeros(np.matmul(z_av, z_av.T).shape)
        for i in range(len(z_hat)):
            st += self.sigmas.Wc[i] * np.dot((z_av[i] - z_hat), (z_av[i] - z_hat).T)
        print("st", st)
        exit()
        return st

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
        mu_a_t_1 = np.array([list(mut_1) + [0, 0, 0, 0]])
        # line 5
        sigma_a_t_1 = self.get_sigma(sigmat_1, Mt, Qt)

        # Generate sigma points
        # line 6
        sigma_points = self.sigmas.sigma_points(mu_a_t_1, sigma_a_t_1)
        
        # Pass sigma points through motion model and compute Gaussian statistics
        # line 7
        khi_x_t_hat = self.pass_points(sigma_points, ut)
        # line 8
        mu_t_hat = self.estimate_mean(khi_x_t_hat)
        # line 9
        sigma_t_hat = self.estimate_sigma(khi_x_t_hat, mu_t_hat)
        # predict step
        # Predict observations at sigma points and compute Gaussian statistics
        # line 10
        z_t_av = self.pass_observations(khi_x_t_hat, sigma_points)
        # line 11
        z_t_hat = self.estimate_obs(z_t_av)
        # line 12 :: uncertainty ellipse
        s_t = self.estimate_uncertainty_ellipse(z_t_av, z_t_hat)
        # line 13
        sigma_t_xz = self.estimate_gain_sigma(khi_x_t_hat, mu_t_hat,  z_t_av, z_t_hat)
        # update step
        # Update mean and covariance
        # line 14
        print(sigma_t_xz.shape)
        print(s_t)
        K_t = np.matmul(sigma_t_xz, np.linalg.inv(s_t))
        exit()
        # line 15
        # line 16
        # line 17

        # line 18
        mut, sigmat, pzt = [0,0,0]
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
    for i in range(25):
        u.append([1, -0.1])
    for i in range(15):
        u.append([1, 0.])
    for i in range(15):
        u.append([1, 0.1])
    return u


def main():
    # INITIALIZATION
    env = Env(corr=False)
    # initial pose estimate
    pose_initial = [100, 100, 0]
    mu = pose_initial
    # initial variance
    sigma = np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ])
    # add starting points for trajectories
    env.add_real(pose_initial)
    env.add_dead(pose_initial)
    env.add_estimate(mu, sigma)
    # motion model
    dt = 1.0
    motion = MotionModels(dt, distribution="normal")
    # localization algorithm
    ukf = UKF(motion, dt)
    ukf.set_qt(env.get_qt())
    ukf.set_obs(env.get_observations)
    # command list
    cmds = commands()
    # correspondences of sys.argv[1] landmarks
    c = [i for i in range(len(env.get_landmarks()))]
    # map
    m = env.get_landmarks()
    # MOVING
    for i, u in enumerate(cmds):
        # Move and observe.
        x_real_prev = env.get_real_pose()
        x_dead_prev = env.get_dead_pose()
        x_real = motion.sample_motion_model_velocity(x_real_prev, u)
        x_dead = motion.sample_motion_model_velocity(x_dead_prev, u, noise=False)
        env.add_real(x_real)
        env.add_dead(x_dead)
        z = env.get_observations(None, False, True)
        mu, sigma, p_t = ukf.localization(mu, sigma, u, z, c)
        env.add_estimate(mu, sigma)
    env.draw()


if __name__ == "__main__":
    main()
