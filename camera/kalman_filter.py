import numpy as np


class KalmanFilterCA3D:

    def __init__(self,
                 q_pos=1e-4,
                 q_vel=1e-4,
                 q_acc=1e-3,
                 r_pos=1.0,
                 P0_scale=10.0):
        self.n = 9
        self.m = 3

        self.q_pos = q_pos
        self.q_vel = q_vel
        self.q_acc = q_acc

        self.x = np.zeros((self.n, 1))
        self.P = P0_scale * np.eye(self.n)

        self.H = np.hstack([np.eye(3), np.zeros((3, 6))])
        self.R = r_pos * np.eye(3)

        self.F = np.eye(self.n)
        self.Q = np.zeros((self.n, self.n))

    @staticmethod
    def build_F(dt):
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        F = np.block([
            [I3, dt * I3, 0.5 * (dt ** 2) * I3],
            [Z3, I3, dt * I3],
            [Z3, Z3, I3]
        ])
        return F

    def build_Q(self, dt):
        q_p = self.q_pos * dt ** 4
        q_v = self.q_vel * dt ** 2
        q_a = self.q_acc * dt

        Q = np.block([
            [q_p * np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), q_v * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), q_a * np.eye(3)]
        ])
        return Q

    def set_measurement_model(self, H, R):
        self.H = H
        self.R = R
        self.m = H.shape[0]

    def predict(self, dt):
        dt = float(dt)

        self.F = self.build_F(dt)
        self.Q = self.build_Q(dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape(self.m, 1)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.x

        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        x = self.x.flatten()
        return {
            "pos": x[0:3].copy(),
            "vel": x[3:6].copy(),
            "acc": x[6:9].copy()
        }
