import numpy as np
from controllers.controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        model1 = ManiuplatorModel(Tp, 0.1, 0.05)
        model2 = ManiuplatorModel(Tp, 0.01, 0.01)
        model3 = ManiuplatorModel(Tp, 1.0, 0.3)
        self.models = [model1, model2, model3]
        self.i = 0
        self.K_d = 1
        self.K_p = 5
        self.u_prev = np.zeros((2, 1))
        self.x_prev = [0, 0, 0, 0]

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q = self.x_prev[:2]
        q_dot = self.x_prev[2:]
        errors_ = []
        for i, model in enumerate(self.models):
            x_ddot = np.linalg.inv(model.M(self.x_prev)) @ (np.array(self.u_prev).reshape((2, 1)) - model.C(self.x_prev) @ np.array(q_dot).reshape(2, 1))
            x_dot = np.array(q_dot).reshape((2, 1)) + x_ddot * model.Tp
            x1 = np.concatenate((np.array(q).reshape((2, 1)) + x_dot * model.Tp, x_dot))

            error = np.array([x]).reshape((4, 1)) - x1
            errors_.append(np.linalg.norm(error))
        # print(errors_)
        self.i = np.argmin(errors_)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        e = q_r - q
        e_dot = q_r_dot - q_dot
        v = q_r_ddot + self.K_p * e + self.K_d * e_dot
        # print(self.i)
        # v = q_r_ddot # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.u_prev = u
        self.x_prev = x
        return u
