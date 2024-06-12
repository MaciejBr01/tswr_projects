import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.K_p = Kp
        self.K_d = Kd
        self.L = np.array([[3*p[0], 0],
                            [0, 3*p[1]],
                            [3*p[0]**2, 0],
                            [0, 3*p[1]**2],
                            [p[0]**3, 0],
                            [0, p[1]**3]])
        W = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0]])
        A = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        B = np.zeros((6, 2))
        self.A1 = A
        self.B1 = B
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot])
        M_inv = np.linalg.inv(self.model.M(x))
        self.A1[2:4, 2:4] = -M_inv @ self.model.C(x)
        self.B1[2:4, :] = M_inv
        self.eso.A = self.A1
        self.eso.B = self.B1

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        q = np.array([[q1], [q2]])
        z_hat = self.eso.get_state()
        q_hat = z_hat[0:2]
        q_dot_hat = z_hat[2:4]
        f = z_hat[4:6]
        e = q_d - q_hat
        e_dot = q_d_dot - q_dot_hat
        v = q_d_ddot + np.float64(self.K_p) @ e + np.float64(self.K_d) @ e_dot
        u = self.model.M(x) @ (v - f) + self.model.C(x) @ q_dot_hat
        self.update_params(q_hat, q_dot_hat)
        self.eso.update(q, u.reshape(len(u), 1))

        return u


