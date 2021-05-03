#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import random

class Env:
    def __init__(self, x, y, th, landmarks_num, qt):
        # observation noise
        self.qt = qt
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
        # path points
        self.points = []
        # lines from robot to landmark
        self.lines = []
        # measurment points
        self.measurment_points = []
        # landmarks
        self.landmarks = []
        for i in range(landmarks_num):
            x_ = np.clip(random.randint(10, self.size[0] - 10), 0, 700)
            y_ = np.clip(random.randint(10, self.size[1] - 10), 600, 700)
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

    def draw_lines_from_robot(self, img):
        for line in self.lines:
            cv2.line(img, line[0], line[1], (55, 55, 0), 1)
        self.lines = []
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
        for j in self.measurment_points:
            img = cv2.circle(img, j, 1, (100, 0, 0), -1)
        return img

    def draw_landmarks(self, img):
        for i, landmark in enumerate(self.landmarks):
            if i == 0:
                cv2.circle(img, landmark, 3, (255, 0, 255), -1)
            else:
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
        # add a path point
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

    def get_observations(self):
        p = self.robot_pose
        z = []
        # precision of distance measurement
        for i, lm in enumerate(self.landmarks):
            r = ((lm[0]-p[0])**2+(lm[1]-p[1])**2) ** 0.5 + np.random.normal(0, self.qt[0][0]**0.5)
            phi = np.arctan2((lm[1]-p[1]), (lm[0]-p[0])) + np.random.normal(0, self.qt[1][1]**0.5)
            point = (
                int(lm[0] - r * np.cos(phi)),
                int(lm[1] - r * np.sin(phi))
            )
            self.measurment_points.append(point)
            self.lines.append(
                [
                    (
                        int(p[0]),
                        int(p[1])
                    ),
                    (
                        int(p[0] + r * np.cos(phi)),
                        int(p[1] + r * np.sin(phi))
                    )
                ]
            )
            phi = phi - p[2]  # -theta
            z.append([r, phi, i])
        return z

    def set_pose(self, pose):
        self.robot_pose = [int(pose[0]), int(pose[1]), pose[2]]

    def draw_step(self, i, l, mu, sigma):
        self.add_point(mu[:2])
        scene = self.get_empty_scene()
        scene = self.draw_landmarks(scene)
        scene = self.draw_robot(scene)
        scene = self.draw_lines_from_robot(scene)
        scene = self.draw_position_estimation(scene, mu, sigma)
        scene = self.draw_label(scene, i, l)
        scene = self.draw_points(scene)
        cv2.imshow('scene', scene)
        time.sleep(0.01)
        cv2.waitKey(1)