#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm, triang
import random



class MotionModels:
    def __init__(self, dt=1, distribution="normal"):
        self.dt = dt
        self.distribution = distribution
        # noises
        self.a = [0.0001 for i in range(6)]

    def get_alpha(self):
        return self.a

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

    def circular_movement(self, x, u):
        if self.distribution == "normal":
            sample = self.sample_normal_distribution
        elif self.distribution == "triangular":
            sample = self.sample_triangular_distribution
        else:
            raise(NotImplemented())
        v_hat = u[0] + sample(self.a[0] * u[0]**2 + self.a[1] * u[1]**2)
        w_hat = u[1] + sample(self.a[2] * u[0] ** 2 + self.a[3] * u[1] ** 2)
        gamma_hat = sample(self.a[4] * u[0] ** 2 + self.a[5] * u[1] ** 2)
        x_ = [0, 0, 0]
        # to prevent division by zero
        if w_hat == 0.:
            w_hat = 0.0001
        x_[0] = x[0] - v_hat / w_hat * np.sin(x[2]) + v_hat / w_hat * np.sin(x[2] + w_hat * self.dt)
        x_[1] = x[1] + v_hat / w_hat * np.cos(x[2]) - v_hat / w_hat * np.cos(x[2] + w_hat * self.dt)
        x_[2] = x[2] + w_hat * self.dt + gamma_hat * self.dt
        #if x_[2] > np.pi:
        #    x_[2] = -np.pi + (x_[2] - np.pi)
        #if x_[2] < -np.pi:
        #    x_[2] = np.pi + (x_[2] + np.pi)
        return x_

    def linear_movement(self, x, u):
        if self.distribution == "normal":
            sample = self.sample_normal_distribution
        elif self.distribution == "triangular":
            sample = self.sample_triangular_distribution
        else:
            raise(NotImplemented())
        v_hat = u[0] + sample(self.a[0] * u[0]**2)
        gamma_hat = sample(self.a[4] * u[0] ** 2 + self.a[5] * u[1] ** 2)

        x_ = [
            x[0] - v_hat * np.sin(x[2]),
            x[1] + v_hat * np.cos(x[2]),
            x[2] + gamma_hat * self.dt
        ]
        return x_

    def circular_movement_jacobian(self, x, u):
        theta = x[2]
        vt, wt = u
        return np.array([
            [1, 0, -vt / wt * np.cos(theta) + vt / wt * np.cos(theta + wt * self.dt)],
            [0, 1, -vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * self.dt)],
            [0, 0, 1],
        ])

    def linear_movement_jacobian(self, x, u):
        vt, wt = u
        theta = x[2]
        return np.array([
            [1, 0, vt * np.cos(theta)],
            [0, 1, vt * np.sin(theta)],
            [0, 0, 1],
        ])

    def get_jacobian(self, x, u):
        if u[1] != 0:
            return self.circular_movement_jacobian(x, u)
        else:
            return self.linear_movement_jacobian(x, u)

    def sample_motion_model_velocity(self, x, u):
        if u[1] != 0:
            return self.circular_movement(u, x)
        else:
            return self.linear_movement(u, x)

    def sample_normal_distribution(self, b2):
        b = b2 ** 0.5
        return 0.5 * sum([random.uniform(-b, b) for _ in range(12)])

    def sample_triangular_distribution(self, b2):
        b = b2 ** 0.5
        return 6 ** 0.5 * 0.5 * (random.uniform(-b, b) + random.uniform(-b, b))