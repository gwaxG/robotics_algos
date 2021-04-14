#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class LikelihoodFields:
    def __init__(self, zmax=3.0, xksens=0., yksens=0., thsens=0.):
        self.zmax = zmax
        self.xksens = xksens
        self.yksens = yksens
        self.thsens = thsens

    def likelihood_field_range_finder_model(self, z, x_pose, m):
        """
        :param z: list of observations [[distance, angle],..]
        :param x: robot pose [x, y, theta]
        :param m: map h*w [[], []...]
        :return:
        """
        q = 1
        x, y, th = x_pose
        for zk, tk in z:
            if zk != self.zmax:
                x_ = x + self.xksens * np.cos(th) - self.yksens * np.sin(th) + zk * np.cos(th+self.thsens)
                y_ = y + self.yksens * np.cos(th) + self.xksens * np.sin(th) + zk * np.sin(th + self.thsens)
                dist = self.get_closest()
                q *= (z_hit * self.prob(dist, sigma_hit) + z_random/z_max)
