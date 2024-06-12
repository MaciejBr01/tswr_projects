import numpy as np
from models.manipulator_model import ManiuplatorModel
from controllers.controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp,1,0.05)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.

        """
        q1, q2, q1_dot, q2_dot = x
        q = [q1, q2]
        q_dot = [q1_dot, q2_dot]

        e = -np.array(q) + q_r
        e_dot = -np.array(q_dot) + q_r_dot

        K_d = np.array([[1, 0],
                        [0, 1]])
        K_p = np.array([[5, 0],
                       [0, 25]])
        v = (q_r_ddot
             + K_d @ e_dot + K_p @ e)
        # v = q_r_ddot

        tau = self.model.M(x) @ v + self.model.C(x) @ q_r_dot

        return tau
