#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import random
import time
from motion import MotionModels
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot


class EKF:
    def __init__(self, motion, dt=1):
        self.motion = motion
        self.alpha = motion.get_alpha()
        self.dt = dt
        self.qt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def localization_with_known_correspondences(
            self,
            mut_1,
            sigmat_1,
            ut,
            zt,
            ct,
            m
    ):
        """
        The extended Kalman filter (EKF) localization algorithm, formulated here
        for a feature-based map and a robot equipped with sensors for measuring range and
        bearing. This version assumes knowledge of the exact correspondences (p. 204, Probabilistic Robotics)
        The _t denotes a time step t, its absence means the time step t-1.
        :param mut_1: pose at previous step
        :param sigmat_1: variance at previous step
        :param ut: taken action
        :param zt: got observation
        :param ct: correspondences of landmarks
        :param m: map (knowledge about coordinates of landmarks)
        :return:
        """
        dt = self.dt
        vt, wt = ut
        theta = mut_1[2]
        Gt = self.motion.get_jacobian(mut_1, ut)
        if wt != 0:
            Vt = np.array([
                [
                    (-np.sin(theta) + np.sin(theta+wt*dt)) / wt,
                    (np.sin(theta) - np.sin(theta+wt*dt)) * vt / wt ** 2 + np.cos(theta + wt * dt) * vt * dt / wt
                ],
                [
                    (np.cos(theta) - np.cos(theta + wt * dt)) / wt,
                    - (np.cos(theta) - np.cos(theta + wt * dt)) * vt / wt ** 2 + np.sin(theta + wt * dt) * vt * dt / wt
                ],
                [0, dt]
            ])
        else:
            Vt = np.array([
                [
                    np.cos(theta),
                    0
                ],
                [
                    0,
                    np.sin(theta)
                ],
                [0, 0]
            ])
        Mt = np.array([
            [self.alpha[0] * vt**2 + self.alpha[1] * wt ** 2, 0],
            [0, self.alpha[2] * vt**2 + self.alpha[3] * wt ** 2]
        ])
        if wt != 0:
            mut_hat = mut_1 + np.array([
                -vt / wt * np.sin(theta) + vt / wt * np.sin(theta + wt * dt),
                vt / wt * np.cos(theta) - vt / wt * np.cos(theta + wt * dt),
                wt * dt
            ])
        else:
            mut_hat = mut_1 + np.array([
                vt * np.cos(theta),
                vt * np.sin(theta),
                0
            ])
        sigmat_hat = Gt @ sigmat_1 @ Gt.T + Vt @ Mt @ Vt.T
        Qt = self.qt
        z_array = []
        S_array = []
        for i, zti in enumerate(zt):
            j = ct[i]
            mjx = m[j][0]
            mjy = m[j][1]
            q = (mjx - mut_hat[0]) ** 2 + (mjy - mut_hat[1]) ** 2
            zti_hat = np.array([q ** 0.5, np.arctan2(mjy - mut_hat[1], mjx - mut_hat[0]), i]).T
            Hti = np.array([
                [-(mjx - mut_hat[0])/q ** 0.5, -(mjy - mut_hat[1])/q ** 0.5, 0],
                [(mjy - mut_hat[1])/q ** 0.5, -(mjx - mut_hat[0])/q ** 0.5, -1],
                [0, 0, 0],
            ])
            Sti = Hti @ sigmat_hat @ Hti.T + Qt
            Kti = sigmat_hat @ Hti.T @ np.linalg.inv(Sti)
            mut_hat = mut_hat + Kti @ (zti - zti_hat)
            sigmat_hat = (np.eye(3) - Kti @ Hti) @ sigmat_hat
            z_array.append(zti_hat)
            S_array.append(Sti)
        mut = mut_hat
        sigmat = sigmat_hat
        pzt = 1
        for i in range(len(zt)):
            pzt *= np.linalg.det(2 * np.pi * S_array[i]) ** 0.5 * np.exp(-0.5 * (zt[i] - z_array[i]).T @ S_array[i] @ (zt[i] - z_array[i]))
        return mut, sigmat, pzt

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
        # path points
        self.points = []
        # landmarks
        self.landmarks = []
        for i in range(3):
            x_ = random.randint(10, self.size[0] - 10)
            y_ = random.randint(10, self.size[1] - 10)
            self.landmarks.append((x_, y_))

    def get_landmarks(self):
        return self.landmarks

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

    def draw_points(self, img):
        for p in self.points:
            img = cv2.circle(img, p, 2, (100, 100, 100), -1)
        return img

    def draw_landmarks(self, img):
        for landmark in self.landmarks:
            cv2.circle(img, landmark, 3, (255, 0, 0), -1)
        return img

    def draw_position_estimation(self, img, x, sigma):
        radius = 15
        center = (int(x[0]), int(x[1]))
        color = (0, 0, 0)
        cv2.circle(img, center, radius, color, -1)
        end = (
            int(center[0] + radius * np.cos(x[2])),
            int(center[1] + radius * np.sin(x[2]))
        )
        cv2.line(img, center, end, (255, 255, 255), 1)

        # covariance ellipse
        a = sigma[0][0]
        b = sigma[0][1]
        c = sigma[1][1]
        l1 = (a + c) / 2 + (0.25 * (a - c)**2 + b ** 2) ** 0.5
        l2 = (a + c) / 2 - (0.25 * (a - c) ** 2 + b ** 2) ** 0.5
        if b == 0 and a >= c:
            angle = 0
        elif b == 0 and a < c:
            angle = np.pi / 2
        else:
            angle = np.arctan2(l1-a, b)
        print(l1)
        img = cv2.ellipse(
            img=img,
            center=center,
            axes=(int(l1), int(l2)),
            angle=angle*180/3.14,
            startAngle=0.,
            endAngle=360.,
            color=(0, 0, 255),
            thickness=1
        )
        return img

    def add_point(self, p):
        self.points.append((int(p[0]), int(p[1])))

    def draw_label(self, img, i, l):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (550, 50)
        fontScale = 1
        fontColor = (255, 0, 0)
        lineType = 2

        cv2.putText(img, str(i)+"/"+str(l),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return img

    def get_observations(self, qt):
        p = self.robot_pose
        z = []
        # precision of distance measurement
        for i, lm in enumerate(self.landmarks):
            r = ((lm[0]-p[0])**2+(lm[1]-p[1])**2) ** 0.5 + np.random.normal(0, qt[0][0]**0.5)
            phi = np.arctan2((lm[1]-p[1]), (lm[0]-p[0])) - p[2] + np.random.normal(0, qt[1][1]**0.5)
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
    # initialization
    initial_pose = [100, 100, 0]
    # pose estimate
    mu = initial_pose
    # real pose
    x = initial_pose
    sigma = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.1],
    ])
    # environment
    env = Env(*initial_pose)
    # motion model
    motion = MotionModels(dt=1, distribution="normal")
    # localization algorithm
    ekf = EKF(motion)
    # command list
    cmds = commands()
    # correspondences of 3 landmarks
    c = [0, 1, 2]
    # map
    m = env.get_landmarks()
    # iterate over command list
    for i, u in enumerate(cmds):
        # Move and observe.
        qt = ekf.get_qt()
        x = motion.sample_motion_model_velocity(u, x)
        env.set_pose(x)
        z = env.get_observations(qt)
        # estimate pose
        mu, sigma, p_t = ekf.localization_with_known_correspondences(mu, sigma, u, z, c, m)
        env.add_point(mu[:2])
        # Visualize
        scene = env.get_empty_scene()
        scene = env.draw_landmarks(scene)
        scene = env.draw_robot(scene)
        scene = env.draw_position_estimation(scene, mu, sigma)
        scene = env.draw_label(scene, i, len(cmds))
        scene = env.draw_points(scene)
        cv2.imshow('scene', scene)
        cv2.waitKey(1)
        # print(((x[0]-mu[0])**2 + (x[1]-mu[1])**2)**0.5)
        time.sleep(0.01)
        # reassigning current state


if __name__ == "__main__":
    run()
