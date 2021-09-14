#!/usr/bin/env python3

""" TODO: docstring """

import numpy as np
# np.random.seed(42)

import lie_algebra as lie
from landmark_detection import landmark_detection

landmarks = np.array([
    [1.0, 5.0, 0.0],
    # [1.0, 10.0, 0.0],
    # [1.0, -5.0, 0.0],
    # [1.0, -10.0, 0.0],
    [-5.0, 5.0, 0.0],
    [5.0, 5.0, 0.0],
    # [5.0, -5.0, 0.0],
    # [-5.0, -5.0, 0.0],

    # [0.0, 0.0, 0.0],
    # [-2.0, 2.0, 0.0],
    # [2.0, 2.0, 0.0],
    # [2.0, -2.0, 0.0],
    # [-2.0, -2.0, 0.0],
    [-2.0, 8.0, 0.0],
])
# Add randomness to landmark location to avoid alignments
landmarks[:, :2] += np.random.normal(0.0, 1.0, (landmarks.shape[0], 2))


def compute_gdop(measurements):
    """ TODO: docstring """
    # Geometry (LOS) matrix
    H = np.empty((measurements.shape[0], 4))
    for m in range(measurements.shape[0]):
        dx = - measurements[m, 0, 3]
        dy = - measurements[m, 1, 3]
        dz = - measurements[m, 2, 3]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        H[m, 0] = dx / r
        H[m, 1] = dy / r
        H[m, 2] = dz / r
        H[m, 3] = 1.0
    H_t = H.transpose()
    print("H:\n{}".format(H))
    print("H_t:\n{}".format(H_t))
    print("H·H_t:\n{}".format(H.dot(H_t)))

    # Matrix inversion method
    # Covariance matrix
    Q = np.linalg.inv(H.dot(H_t))    
    print("Q:\n{}".format(Q))
    gdop = np.sqrt(np.trace(Q))

    # Weird method
    # B = H.dot(H_t)
    # a = (B[0,1]*B[2,3] + B[0,2]*B[1,3] - B[0,3]*B[1,2])**2 - 4*B[0,1]*B[2,3]*B[0,2]*B[1,3]
    # b = 16 - 4 * (B[0,1]**2 + B[0,2]**2 + B[0,3]**2 + B[1,2]**2 + B[1,3]**2 + B[2, 3]**2)
    # c = 2 * (B[0,1] * (B[0,2]*B[1,2] + B[0,3]*B[1,3]) + B[2,3] * (B[0,2]*B[0,3] + B[1,2]*B[1,3]))

    # print(a)
    # print(b)
    # print(c)

    # print((16 + b + c))
    # print((a + b + 2*c))

    # gdop = np.sqrt((16 + b + c) / (a + b + 2*c))

    print("gdop:\n{}".format(gdop))
    return gdop

def compute_gdop_2d(measurements):
    """ TODO: docstring """
    # Geometry (LOS) matrix
    H = np.empty((measurements.shape[0], 3))
    for m in range(measurements.shape[0]):
        dx = - measurements[m, 0, 3]
        dy = - measurements[m, 1, 3]
        r = np.sqrt(dx**2 + dy**2)
        H[m, 0] = dx / r
        H[m, 1] = dy / r
        H[m, 2] = 1.0
    H_t = H.transpose()
    print("H:\n{}".format(H))
    print("H_t:\n{}".format(H_t))
    # print("H·H_t:\n{}".format(H.dot(H_t)))
    print("H_t·H:\n{}".format(H_t.dot(H)))

    # Matrix inversion method
    # Covariance matrix
    Q = np.linalg.inv(H.dot(H_t))    
    print("Q:\n{}".format(Q))
    gdop = np.sqrt(np.trace(Q))

    print("gdop:\n{}".format(gdop))
    return gdop

def compute_wgdop(measurements, measurement_covs):
    """ TODO: docstring """
    pass


def compute_crlb(measurements, measurement_covs):
    cov_inv = np.linalg.inv(measurement_covs[0])
    G = np.empty((measurements.shape[0], 2))
    dx_0 = measurements[0, 0, 3]
    dy_0 = measurements[0, 1, 3]
    r_0 = np.sqrt(dx_0**2 + dy_0**2)
    for m in range(measurements.shape[0] - 1):
        dx = measurements[m + 1, 0, 3]
        dy = measurements[m + 1, 1, 3]
        r = np.sqrt(dx**2 + dy**2)
        G[m, 0] = (dx_0 / r_0) - (dx / r)
        G[m, 1] = (dy_0 / r_0) - (dy / r)
    G_t = G.transpose()
    print("G:\n{}".format(G))
    print("G_t:\n{}".format(G_t))
    FIM = G_t.dot(cov_inv).dot(G)
    print("FIM = G_t·Q·G:\n{}".format(FIM))
    crlb = np.linalg.inv(FIM)
    print("crlb:\n{}".format(crlb))
    return crlb




if __name__ == '__main__':
    X = np.array([2.0, 3.0, 0.0])
    X[:2] += np.random.normal(0.0, 0.2, 2)
    print("X: {}".format(X))
    print("Landmarks:\n{}".format(landmarks))

    # Get the landmark measurements
    X_se3 = lie.se3(t=[X[0], X[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, X[2]]))
    measurements, measurement_covs = landmark_detection(X_se3, landmarks, std=0.1)
    print("measurements:\n{}".format(measurements))
    print("measurement_covs:\n{}".format(measurement_covs))
    gdop = compute_gdop_2d(measurements)
    print(measurement_covs[0])
    # crlb = compute_crlb(measurements, measurement_covs)
