#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
from scipy.stats import norm, triang


class MotionModels:
    def __init__(self, dt=1, distribution="normal"):
        self.dt = dt
        self.distribution = distribution
        self.a = [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ]

    def motion_model_velocity(self, x_, u, x):
        mu = 0.5
        mu *= ((x[0] - x_[0]) * np.cos(x[2]) + (x[1] - x_[1]) * np.sin(x[2]))
        mu /= ((x[1] - x_[1]) * np.cos(x[2]) + (x[0] - x_[0]) * np.sin(x[2]))
        x_star = (x[0] + x_[0]) * 0.5 + mu * (x[1] - x_[1])
        y_star = (x[1] + x_[1]) * 0.5 + mu * (x_[0] - x[0])
        r_star = ((x[0] - x_star) ** 2 + (x[1] - y_star) ** 2) ** 0.5
        dtheta = np.atan2(x_[1] - y_star, x_[0] - x_star) - np.atan2(x[1] - y_star, x[0] - x_star)
        v_hat = dtheta / self.dt * r_star
        w_hat = dtheta / self.dt
        gamma_hat = (x_[2] - x[2]) / self.dt - w_hat
        if self.distribution == "normal":
            prob = self.prob_normal_distribution
        elif self.distribution == "triangular":
            prob = self.prob_triangular_distribution
        prob1 = prob(u[0]-v_hat, self.a[0] * u[0]**2 + self.a[1] * u[1]**2)
        prob2 = prob(u[1] - w_hat, self.a[2] * u[0] ** 2 + self.a[3] * u[1] ** 2)
        prob3 = prob(gamma_hat, self.a[4] * u[0] ** 2 + self.a[5] * u[1] ** 2)
        return prob1 * prob2 * prob3

    def prob_normal_distribution(self, a, b2):
        return norm(a, b2).pdf(0)

    def prob_triangular_distribution(self, a, b2):
        return triang(a, b2).pdf(0)

    def sample_motion_model_velocity(self, u, x):
        if self.distribution == "normal":
            sample = self.sample_normal_distribution
        elif self.distribution == "triangular":
            sample = self.sample_triangular_distribution
        v_hat = u[0] + sample(self.a[0] * u[0]**2 + self.a[1] * u[1]**2)
        w_hat = u[1] + sample(self.a[2] * u[0] ** 2 + self.a[3] * u[1] ** 2)
        gamma_hat = sample(self.a[4] * u[0] ** 2 + self.a[5] * u[1] ** 2)
        x_ = [0, 0, 0]
        x_[0] = x[0] - v_hat / w_hat * np.sin(x[2]) + v_hat / w_hat * np.sin(x[2] + w_hat * self.dt)
        x_[1] = x[1] + v_hat / w_hat * np.cos(x[2]) - v_hat / w_hat * np.cos(x[2] + w_hat * self.dt)
        x_[2] = x[2] + w_hat * self.dt + gamma_hat * self.dt
        return x_

    def sample_normal_distribution(self, b2):
        b = b2 ** 0.5
        return 0.5 * sum([random.uniform(-b, b) for _ in range(12)])

    def sample_triangular_distribution(self, b2):
        b = b2 ** 0.5
        return 6 ** 0.5 * 0.5 * (random.uniform(-b, b) + random.uniform(-b, b))

    def motion_model_odometry(self, x_, u, x):
        # u = [x_hat, x__hat]
        drot1 = np.atan2(u[1][1] - u[0][1], u[1][0] - u[0][0]) - u[0][2]
        dtrans = ((u[0][0] - u[1][0])**2 + (u[0][1] - u[1][1])**2) ** 0.5
        drot2 = u[1][2] - u[0][2] - drot1

        drot1_hat = np.atan2(x_[1] - x[1], x_[0] - x[0]) - x[2]
        dtrans_hat = ((x[0] - x_[0]) ** 2 + (x[1] - x_[1]) ** 2) ** 0.5
        drot2_hat = x_[2] - x[2] - drot1_hat

        if self.distribution == "normal":
            prob = self.prob_normal_distribution
        elif self.distribution == "triangular":
            prob = self.prob_triangular_distribution
        p1 = prob(drot1 - drot1_hat, self.a[0] * drot1_hat ** 2 + self.a[1] * dtrans_hat ** 2)
        p2 = prob(dtrans - dtrans_hat, self.a[2] * dtrans_hat ** 2 + self.a[3] * (drot1_hat ** 2 + drot2_hat ** 2))
        p3 = prob(drot2 - drot2_hat, self.a[0] * drot2_hat ** 2 + self.a[1] * dtrans_hat ** 2)
        return p1 * p2 * p3

    def sample_motion_model_odometry(self, u, x):
        drot1 = np.atan2(u[1][1] - u[0][1], u[1][0] - u[0][0]) - u[0][2]
        dtrans = ((u[0][0] - u[1][0])**2 + (u[0][1] - u[1][1])**2) ** 0.5
        drot2 = u[1][2] - u[0][2] - drot1

        if self.distribution == "normal":
            sample = self.sample_normal_distribution
        elif self.distribution == "triangular":
            sample = self.sample_triangular_distribution

        drot1_hat = drot1 - sample(self.a[0]*drot1**2+self.a[1]*dtrans**2)
        dtrans_hat = dtrans - sample(self.a[2]*dtrans**2+self.a[3]*drot1**2+self.a[3]*drot2**2)
        drot2_hat = drot2 - sample(self.a[0]*drot2**2+self.a[1]*dtrans**2)

        x_ = [0, 0, 0]
        x_[0] = x[0] + dtrans_hat * np.cos(x[2] + drot1_hat)
        x_[1] = x[1] + dtrans_hat * np.sin(x[2] + drot1_hat)
        x_[2] = x[2] + drot1_hat + drot2_hat
        return x_

    

