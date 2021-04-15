#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import random
import time
from motion import MotionModels
import numpy as np


class EKF:
    def __init__(self):
        self.qt = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]])

    def localization_with_known_correspondences(
            self,
            mu,
            sigma,
            u_,
            z_,
            c_,
            m
    ):
        """
        The extended Kalman filter (EKF) localization algorithm, formulated here
        for a feature-based map and a robot equipped with sensors for measuring range and
        bearing. This version assumes knowledge of the exact correspondences (p. 204, Probabilistic Robotics)
        The underscore _ denotes a time step t, its absence means the time step t-1.
        :param mu: pose at previous step
        :param sigma: variance at previous step
        :param u_: taken action
        :param z_: got observation
        :param c_: correspondences of landmarks
        :param m: map (knowledge about coordinates of landmarks)
        :return:
        """
        mu_ = []
        sigma_ = []
        pz_ = []
        return mu_, sigma_, pz_

    def get_qt(self):
        return self.qt

class Env:
    def __init__(self, x, y, th):
        # size of the field
        self.size = [700, 700]
        # starting robot pose
        initial_pose = [x, y, th]
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
            x_ = random.randint(10, self.size[0] - 10)
            y_ = random.randint(10, self.size[1] - 10)
            self.landmarks.append((x_, y_))

    def get_empty_scene(self):
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
        end = (
            int(center[0] + radius * np.cos(self.robot_pose[2])),
            int(center[1] + radius * np.sin(self.robot_pose[2]))
        )
        cv2.line(img, center, end, (255, 255, 255), 4)
        return img

    def draw_path(self, img, path, color):
        path = np.array(path).T
        curve = np.column_stack((path[0].astype(np.int32), path[1].astype(np.int32)))
        cv2.polylines(img, [curve], False, color)
        return img

    def draw_landmarks(self, img):
        for landmark in self.landmarks:
            cv2.circle(img, landmark, 3, (255, 0, 0), -1)
        return img

    def draw_position_estimation(self):
        # draw ellipse around the robot
        pass

    def get_observations(self, p, qt):
        z = []
        for i, lm in enumerate(self.landmarks):
            r = ((lm[0]-p[0])**2+(lm[1]-p[1])**2) ** 0.5 + np.random.normal(0, qt[0][0])
            phi = np.atan2((lm[1]-p[1]), (lm[0]-p[0])) - p[2] + np.random.normal(0, qt[1][1])
            z.append([r, phi, i])
        return z

    def set_pose(self, pose):
        self.robot_pose = [int(pose[0]), int(pose[1]), pose[2]]

def commands():
    """
    :return: list of commands
    """
    u = []
    for i in range(150):
        u.append([1, 0.01])
    for i in range(300):
        u.append([1, 0])
    for i in range(150):
        u.append([1, -0.01])
    return u


def run():
    initial_pose = [100, 100, 0]
    # robot pose x_t-1 and robot pose x_t
    x = initial_pose
    x_ = initial_pose
    # environment
    env = Env(*initial_pose)
    # localization algorithm
    ekf = EKF()
    # motion model
    motion = MotionModels(dt=1, distribution="normal")
    # do for every command vector u
    cmds = commands()
    for i, u in enumerate(cmds):
        # localization
        qt = ekf.get_qt()
        z = env.get_observations(x, qt)
        x_ = motion.sample_motion_model_velocity(u, x)
        env.set_pose(x_)
        # visualization
        scene = env.get_empty_scene()
        scene = env.draw_landmarks(scene)
        scene = env.draw_robot(scene)
        cv2.imshow('scene', scene)
        cv2.waitKey(1)
        print(f"{i}/{len(cmds)}  moved  to  {[int(x_[0]), int(x_[1])]} under {u} from {[int(x[0]), int(x[1])]}")
        time.sleep(0.01)
        # reassigning current state
        x = x_


if __name__ == "__main__":
    run()
