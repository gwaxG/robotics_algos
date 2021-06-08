#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Map:
    """
    A cone consists of five beams.
    """
    def __init__(self):
        self.width = 21
        self.height = 21
        self.map = np.zeros((self.width, self.height))
        self.map_discovered = np.zeros((self.width, self.height))
        self.map[:, 0] = 1
        self.map[:, self.width-1] = 1
        self.map[0, :] = 1
        self.map[self.height-1, :] = 1
        self.x = 10
        self.y = 10
        self.theta = 0.

    def rotate(self):
        self.theta += np.pi / 6
        if self.theta >= 2 * np.pi:
            self.theta = 0.

    def ray_cast(self, angle):
        x = self.x
        y = self.y
        d = 1 if 0 <= angle <= np.pi / 2 or 3.0 / 2 * np.pi <= angle <= 2 * np.pi else -1
        while 0 <= x < self.width - 1 and 0 <= y < self.height - 1:
            x += d * np.cos(angle)
            y += d * np.sin(angle)
        return int(x), int(y)

    def measure(self):
        results = []
        # 5 cones
        n = 5
        delta = np.pi / 6 / n
        angle = self.theta + delta
        angles = []
        # ray casting
        for i in range(n):
            angles.append(angle)
            angle += delta
        for angle in angles:
            x, y = self.ray_cast(angle)
            dist = ((x - self.x)**2 + (y - self.y)**2) ** 0.5
            results.append([dist, angle])
        return results

    def draw(self):
        pass


def occupancy_grid_mapping(l, x, z):
    pass


def inverse_measurement_model(m, x, z):
    pass


def main():
    """
    The robot is located at the center of the square-shaped room of the size 51x51 (25x25).
    It rotates 12 times by 30 degrees.
    The FOV cone is 30 degrees.
    """
    m = Map()
    print(m.measure())

if __name__ == "__main__":
    main()
