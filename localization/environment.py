#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class Env:
    def __init__(self, corr=True):
        self.correspondences = corr
        # landmarks
        self.landmarks = [
            [0, 0],
            [200, 100],
            [0, 200],
        ]
        # estimated movement
        self.estimation = []
        # variance
        self.variance = []
        self.n_ellipses = 20
        # real movement with noise
        self.real = []
        # pure movement
        self.dead = []
        # observation noise
        self.qt = np.diag([
            10.,
            np.deg2rad(2.0),  # variance of yaw angle
            1
        ]) ** 2
        if not self.correspondences:
            self.qt = self.qt[:2, :2]

    def get_real_pose(self):
        return self.real[-1]

    def get_dead_pose(self):
        return self.dead[-1]

    def add_real(self, x):
        self.real.append(x)

    def add_dead(self, x):
        self.dead.append(x)

    def add_estimate(self, mu, var):
        self.estimation.append(mu)
        self.variance.append(var)

    def get_landmarks(self):
        return self.landmarks

    def draw(self):
        # Draw trajectories.
        # landmarks
        # x, y = np.array(self.landmarks).T
        # plt.scatter(x, y, c="k", label="landmarks")
        # dead
        x, y, _ = np.array(self.dead).T
        plt.scatter(x, y, s=3.,c="b", label="Dead reckoning")
        # real
        x, y, _ = np.array(self.real).T
        plt.scatter(x, y, s=3., c="g", label="Real movement")
        # estimate
        x, y, _ = np.array(self.estimation).T
        plt.scatter(x, y, s=3., c="r", label="Filtered movement")
        plt.xlabel("X, pixels")
        plt.ylabel("Y, pixels")
        plt.legend(loc='upper left')
        # Draw ellipses
        if len(self.variance) != 0 and self.variance[0] != []:
            t = np.linspace(0, 2 * np.pi, 100)
            x, y, _ = np.array(self.estimation).T
            # Just error prevention if we want to draw more covariance ellipses than we have variances.
            if self.n_ellipses > len(self.estimation):
                self.n_ellipses = len(self.estimation)
            for i in range(0, len(self.estimation)-1, int(len(self.estimation) / self.n_ellipses)):
                x_cent = x[i]  # x-position of the center
                y_cent = y[i]  # y-position of the center
                sigma = self.variance[i]
                a = sigma[0][0]
                b = sigma[0][1]
                c = sigma[1][1]
                l1 = (a + c) / 2 + (0.25 * (a - c) ** 2 + b ** 2) ** 0.5  # radius on the x-axis
                l2 = (a + c) / 2 - (0.25 * (a - c) ** 2 + b ** 2) ** 0.5  # radius on the y-axis
                if b == 0 and a >= c:
                    theta = 0
                elif b == 0 and a < c:
                    theta = np.pi / 2
                else:
                    theta = np.arctan2(l1 - a, b)
                # https://cookierobotics.com/007/
                x_ellipse = x_cent + l1 ** 0.5 * np.cos(theta) * np.cos(t) - l2 ** 0.5 * np.sin(theta) * np.sin(t)
                y_ellipse = y_cent + l1 ** 0.5 * np.sin(theta) * np.cos(t) + l2 ** 0.5 * np.cos(theta) * np.sin(t)
                plt.plot(x_ellipse, y_ellipse, c="k")
        plt.show()

    def get_qt(self):
        return self.qt

    def get_observations(self, pose, corr=True, single=False):
        if pose is None:
            x, y, theta = self.real[-1]
        else:
            x, y, theta = pose
        observations = []
        for i, lm in enumerate(self.landmarks):
            r = ((lm[0]-x)**2+(lm[1]-y)**2) ** 0.5 + np.random.normal(0, self.qt[0][0]**0.5)
            phi = np.arctan2((lm[1]-y), (lm[0]-x)) + np.random.normal(0, self.qt[1][1]**0.5)
            phi = phi - theta
            if corr:
                observations.append([r, phi, i])
            else:
                observations.append([r, phi])
                if single:
                    return observations[0]
        return observations


def main():
    e = Env()
    p0 = [100, 10, 0]
    p1 = [100, 0, 0]
    p2 = [100, -10, 0]
    for i in range(20):
        p = [p0[0] + i, p0[1] + i, 0]
        e.add_real(p)
        p = [p1[0] + i, p1[1] + i, 0]
        e.add_dead(p)
        p = [p2[0] + i, p2[1] + i, 0]
        e.add_estimate(p, [])
    e.draw()


if __name__ == "__main__":
    main()
