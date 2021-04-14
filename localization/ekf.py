#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import random
import time
import numpy as np


class EKF:
    def __init__(self):
        pass


class Env:
    def __init__(self):
        # size of the field
        self.size = [700, 700]
        # starting robot pose
        initial_pose = [50, 650, np.pi / 2]
        # previous robot pose
        self.robot_pose_prev = initial_pose
        # current robot pose
        self.robot_pose = initial_pose
        # command vector
        self.u = [0, 0]
        # true path
        self.path_true = []
        # dead reckoning path
        self.path_dead = []
        # estimated path
        self.path_est = []
        # landmarks
        self.landmarks = []
        for i in range(3):
            x = random.randint(self.size[0] + 10, self.size[1] - 10)
            y = random.randint(self.size[0] + 10, self.size[1] - 10)
            self.landmarks.append((x, y))

    def get_free(self):
        """
        :return: blank field
        """
        img = np.zeros([self.size[0], self.size[1], 3], dtype=np.uint8)
        img[:] = 255
        return img

    def draw_robot(self, img):
        radius = 20
        center = (self.robot_pose[0], self.robot_pose[1])
        color = (0, 0, 255)
        cv2.circle(img, center, radius, color, -1)
        end = [
            center[0] + radius * np.cos(self.robot_pose[2]),
            center[1] + radius * np.sin(self.robot_pose[2])
        ]
        cv2.line(img, center, end, (255, 255, 255), 4)
        return img

    def draw_path(self, img, path, color):
        path = np.array(path).T
        curve = np.column_stack((path[0].astype(np.int32), path[1].astype(np.int32)))
        cv2.polylines(img, [curve], False, color)
        return img

    def draw_landmarks(self, img):
        for landmark in self.landmarks:
            cv2.circle(img, landmark, 10, (255, 0, 0), -1)
        return img

    def draw_position_estimation(self):
        # draw ellipse around the robot
        pass

    def get_distance_landmarks(self):
        pass
        # calculate noised distance to landmarks

    def cmd(self, v, w):
        self.u = [v, w]

    def make_move(self):
        pass
        # calculate new position of the robot
        # estimate

    def observe(self):
        pass
        # calculate observation of landmarks


if __name__ == "__main__":
    e = Env()
    f = EKF()
    while True:
        image = e.get_env_img()
        cv2.imshow('image', image)
        cv2.waitKey(1)
        time.sleep(1)
