#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import random
import time
from sklearn import preprocessing
from environment import Env
from motion import MotionModels
import numpy as np
import math
from scipy.stats import multivariate_normal


class MCL:
    def __init__(self, initial_pose, limits, motion, dt=1.0):
        self.N = 100
        self.motion = motion
        self.dt = dt
        self.qt = None
        self.observation_sampler = None
        self.px = np.array([initial_pose for i in range(self.N)])
        self.pw = np.array([1./self.N for _ in range(self.N)])

    def sample_probability(self, x, z_star):
        prob = 1
        observations = self.observation_sampler(x, corr=False, single=False)
        for i, [obs_, z_mean] in enumerate(zip(observations, z_star)):
            obs = np.array(obs_).T
            prob *= multivariate_normal.pdf(obs, mean=z_mean, cov=self.qt)
        return prob

    def calc_covariance(self, x_est, px, pw):
        """
        calculate covariance matrix
        see ipynb doc
        """
        n = 1.0 / (1.0 - pw @ pw.T)[0][0]
        cov = np.zeros((3, 3))
        for j in range(3):
            for k in range(3):
                cov[j][k] = np.sum([pw[0][i] * (px[i][j] - x_est[j]) * (px[i][k] - x_est[k]) for i in range(self.N)])
        cov *= n
        return np.zeros((3, 3))

    def localization_with_known_correspondences(self, u, z):
        x_pred = []
        w_pred = []
        # Passing
        for m in range(self.N):
            pt = self.px[m]
            xm = self.motion.sample_motion_model_velocity(pt, u)
            wm = self.sample_probability(xm, z)
            x_pred.append(np.array(xm).flatten())
            w_pred.append(wm)
        w_pred = preprocessing.normalize([w_pred])

        # Pose estimation
        x_est = np.zeros((3,))
        for i in range(self.N):
            x_est += self.px[i] * self.pw[i]
        # covariance estimation
        p_est = self.calc_covariance(x_est, np.array(x_pred), w_pred)

        # Resampling
        x_new = []
        n_pred = w_pred * self.N
        n_pred = [round(el) for el in n_pred[0]]
        for m in range(self.N):
            index = n_pred.index(np.max(n_pred))
            n_pred[index] -= 1
            x_new.append(x_pred[index])
        # reassign new particles
        self.px = x_new
        # Reset weights
        self.pw = np.array([1./self.N for _ in range(self.N)])
        return x_est, p_est

    def set_qt(self, qt):
        self.qt = qt

    def set_observation_sampler(self, f):
        self.observation_sampler = f


def commands():
    """
    :return: list of commands
    """
    u = []
    for i in range(20):
        u.append([0.5, -0.1])
    for i in range(1):
        u.append([0.5, 0.])
    for i in range(10):
        u.append([0.5, 0.1])
    return u


def main():
    # INITIALIZATION
    env = Env(corr=False)
    # initial pose estimate
    pose_initial = [25, 25, 0]
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
    limits = [
        env.x_min,
        env.x_max,
        env.y_min,
        env.y_max
    ]
    mcl = MCL(pose_initial, limits, motion, dt)
    mcl.set_observation_sampler(env.get_observations)
    mcl.set_qt(env.get_qt())
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
        z = env.get_observations(None, corr=False, single=False)
        mu, sigma = mcl.localization_with_known_correspondences(u, z)
        env.add_estimate(mu, sigma)
    env.draw()


if __name__ == "__main__":
    main()
