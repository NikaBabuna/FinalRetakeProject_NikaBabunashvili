import numpy as np
from src import config

class PathController:
    def __init__(self, spline, lookahead=None):
        self.spline = spline
        self.lookahead = float(lookahead if lookahead is not None else getattr(config, "LOOKAHEAD", 25.0))
        self.s = 0.0  # for tests/inspection only

    def update(self, robot_pos, robot_vel=None, dt=None):
        """
        Compatible with BOTH styles:

        old style:
            target = update(robot_pos)

        sim style:
            force, target = update(robot_pos, robot_vel, dt)
        """

        robot_pos = np.asarray(robot_pos, dtype=float).reshape(2)

        # 1) Always project onto the spline NOW (do not "lock forward")
        s_closest = float(self.spline.closest_s(robot_pos))
        self.s = s_closest

        # 2) Lookahead target from the projection point
        s_target = min(self.s + self.lookahead, float(self.spline.length))
        target = self.spline.p(s_target)

        # old tests expect only target
        if robot_vel is None:
            return target

        # new tests expect (force, target)
        robot_vel = np.asarray(robot_vel, dtype=float).reshape(2)
        kp = float(getattr(config, "KP_PATH", 1.2))
        kd = float(getattr(config, "KD_PATH", 0.6))

        force = kp * (target - robot_pos) - kd * robot_vel
        return force, target
