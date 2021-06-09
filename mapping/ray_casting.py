#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Map:
    """
    A cone consists of n beams and is of angle degrees.
    """
    def __init__(self):
        self.width = 20
        self.height = 42
        self.n = 5
        self.angle = np.pi / 20
        self.map = np.zeros((self.height, self.width))
        self.map_discovered = np.zeros((self.height, self.width))
        self.map[:, 0] = 1
        self.map[:, self.width-1] = 1
        self.map[0, :] = 1
        self.map[self.height-1, :] = 1
        self.x = int(self.width / 2)
        self.y = int(self.height / 2)
        self.theta = 0
        self.passed = False
        self.cnt = 0

    def rotate(self):
        self.theta += self.angle
        if self.theta >= 2 * np.pi - 0.5 * self.angle:
            self.theta = 0.
            self.passed = True

    def ray_cast(self, angle):
        x = self.x
        y = self.y
        while 0 < x < self.width - 1 and 0 < y < self.height - 1:
            x += np.cos(angle)
            y += np.sin(angle)
        self.cnt += 1
        return round(x), round(y)

    def measure(self):
        results = []
        delta = self.angle / self.n
        angle = self.theta + delta
        angles = []
        # ray casting
        for i in range(self.n):
            angles.append(angle)
            angle += delta

        for angle in angles:
            x, y = self.ray_cast(angle)
            dist = ((x - self.x)**2 + (y - self.y)**2) ** 0.5
            results.append([dist, angle])
        return results

    def draw(self):
        plt.imshow(self.map_discovered, interpolation='none')
        plt.title(f'ray cast mapping in a square room for rotating {self.n} beam {np.rad2deg(self.angle)} cone sensor', fontweight="bold")
        plt.show()

    def produce_map(self):
        self.passed = False
        while not self.passed:
            measurements = self.measure()
            for measure in measurements:
                x = np.clip(0, self.width-1, int(round(self.x + measure[0] * np.cos(measure[1]))))
                y = np.clip(0, self.height-1, int(round(self.y + measure[0] * np.sin(measure[1]))))
                self.map_discovered[y][x] = 1
            self.rotate()


def main():
    """
    The robot is located at the center of the square-shaped room of the size 51x51 (25x25).
    It rotates 12 times by 30 degrees.
    The FOV cone is 30 degrees.
    """
    m = Map()
    m.produce_map()
    m.draw()


if __name__ == "__main__":
    main()
