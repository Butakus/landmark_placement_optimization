#!/usr/bin/env python3

""" TODO: docstring """

import numpy as np

import lie_algebra as lie

# TODO: Find a better way to setup this 
MESAUREMENT_STD = 0.01

def compute_measurement(pose, landmark, std=MESAUREMENT_STD):
    """ Compute the measurement between pose_1 and pose_2.
        Add a gaussian noise in x and y coordinates with the given std.
        Inputs:
            - pose: SE(3) matrix containing the 6DOF position of the origin.
            - landmark: 3D position (x, y, z) of the landmark.
            - std: Standard deviation of the gaussian noise added in x and y axis.
        Output: A tuple, containing the measurement (a 2D array) and the covariance (2x2 matrix).
    """
    # Build measurement covariance matrix
    measurement_cov = np.zeros((2, 2))
    np.fill_diagonal(measurement_cov, std**2)

    # Convert the landmark pose to a SE(3) matrix
    landmark_se3 = lie.se3(t=landmark)

    # Compute measurement
    measurement = lie.relative_se3(pose, landmark_se3)

    # Add gaussian noise
    measurement[:2, 3] += np.random.multivariate_normal(np.zeros(2), measurement_cov)

    return measurement, measurement_cov


def landmark_detection(pose, landmarks, std=MESAUREMENT_STD):
    """ Compute a measurement from the origin pose to each of the landmarks.
        Inputs:
            - pose: SE(3) matrix containing the 6DOF position of the origin.
            - landmarks: numpy matrix containing the 3D position (x, y, z) of each landmark (Nx3).
        Outputs (tuple):
            - Measurements: Array of SE(3) matrices.
            - Measurement covariances: Array of 2x2 matrices.
    """
    measurements = np.zeros((landmarks.shape[0], 4, 4))
    measurement_covs = np.zeros((landmarks.shape[0], 2, 2))

    for l in range(landmarks.shape[0]):
        measurements[l], measurement_covs[l] = compute_measurement(pose, landmarks[l], std)

    return measurements, measurement_covs

if __name__ == '__main__':
    # Test measurement
    print("Measurement test")
    pose = lie.se3(t=[0.0, 0.0, 0.0], r=lie.so3_from_rpy([0.0, 0.0, np.pi/2]))
    landmark_pose = np.array([5.0, -2.0, 0.0])
    print("origin:\n{}".format(pose))
    print("target: {}".format(landmark_pose))
    measurement, measurement_cov = compute_measurement(pose, landmark_pose, 0.1)
    print("measurement: {}".format(measurement))
    print("measurement_cov:\n{}".format(measurement_cov))

    # Test landmark detection
    print("\nLandmark detection test")
    landmarks = np.array([
        [-5.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [5.0, -5.0, 0.0],
        [-5.0, -5.0, 0.0],
    ])
    pose = lie.se3(t=[1.0, 1.0, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
    print("origin: {}".format(pose))
    print("landmarks:\n{}".format(landmarks))
    measurements, measurement_covs = landmark_detection(pose, landmarks)
    print("measurements:\n{}".format(measurements))
    print("measurement_covs:\n{}".format(measurement_covs))