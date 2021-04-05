import matplotlib.pyplot as plt
import numpy as np

"""
The Drawer class draws the goal pose, executed path and the robot pose.
The goal and robot are represented as triangles: the green one relates to the goal, the black one is the robot.
"""


class Drawer:
    def __init__(self, pause):
        self.exec_path = {"x": [], "y": []}
        self.goal = [0., 0., 0.]
        self.robot = [0., 0., 0.]
        self.pause = pause

    def draw_canvas(self):
        plt.axline((-1, -1), (-1, 1))
        plt.axline((1, -1), (1, 1))
        plt.axline((-1, 1), (1, 1))
        plt.axline((-1, -1), (1, -1))

    def set_robot_pose(self, x, y, theta):
        self.robot = [x, y, theta]

    def set_goal_pose(self, x, y, theta):
        self.goal = [x, y, theta]

    def append_path(self, p):
        self.exec_path["x"].append(p[0])
        self.exec_path["y"].append(p[1])

    def draw_actor(self, x, y, t, c):
        r = 0.025
        pts = np.array(
             [
                  [x + r * np.cos(t + np.pi), y + r * np.sin(t + np.pi)],
                  [x + r * np.cos(t), y + r * np.sin(t)],
                  [x + 4 * r * np.cos(t + np.pi / 2), y + 4 * r * np.sin(t + np.pi / 2)],
             ]
        )

        plt.plot((pts[0][0], pts[1][0]), (pts[0][1], pts[1][1]), c)
        plt.plot((pts[1][0], pts[2][0]), (pts[1][1], pts[2][1]), c)
        plt.plot((pts[2][0], pts[0][0]), (pts[2][1], pts[0][1]), c)

    def update(self):
        self.draw_canvas()
        # draw path
        plt.scatter(self.exec_path["x"], self.exec_path["y"], s=2, color='b')
        # draw robot
        self.draw_actor(*self.robot, 'k')
        # draw goal
        self.draw_actor(*self.goal, 'g')
        plt.draw()
        plt.pause(self.pause)
        plt.clf()


if __name__ == "__main__":
    d = Drawer(0.25)
    d.set_goal_pose(0.5, 0.5, np.pi / 2)
    d.set_robot_pose(0., 0., -np.pi / 2)
    for i in range(10):
        d.append_path([i * 0.1, i * 0.1])
        d.set_goal_pose(0.5, 0.5, i*0.1)
        d.set_robot_pose(0.5, 0.0, -i*0.1)
        d.update()
