import numpy as np
from observers.eso import ESO
from controllers.controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.K_p = kp
        self.K_d = kd
        self.model = ManiuplatorModel(Tp, 1, 0.05)

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [self.b],
                      [0]])
        L = np.array([[3*p],
                     [3*p**2],
                     [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)
        self.u_prev = 0
    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.eso.set_B(np.array([[0],
                                [self.b],
                                [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i):
        ### TODO implement ADRC
        q, q_dot = x
        self.eso.update(q, self.u_prev)
        q_hat, q_dot_hat, f = self.eso.get_state()
        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat
        v = q_d_ddot + np.float64(self.K_p) * e + np.float64(self.K_d) * e_dot
        u = (v - f) / self.b
        self.u_prev = u
        x1 = np.pad(x, pad_width=(0, 2), mode='constant', constant_values=0.)
        M_inv = np.linalg.inv(self.model.M(x1))
        self.set_b(M_inv[i, i])

        return u


