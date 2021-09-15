#!/usr/bin/env python3

""" TODO: docstring """

import numpy as np

import lie_algebra as lie

# TODO: Find a better way to setup this
MESAUREMENT_STD = 0.05

MAX_RANGE = 60.0
# MAX_RANGE = np.inf


def distance(pose, landmark):
    """ Compute the euclidean distance between an SE(3) pose and a landmark (3D array) """
    diff = landmark - pose[:3, 3]
    return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

# def distance_2d(pose, landmark):
#     """ Compute the euclidean distance between an SE(3) pose and a landmark (2D array) """
#     diff = landmark - pose[:2, 3]
#     return np.sqrt(diff[0]**2 + diff[1]**2)


def increase_range_error(std, distance):
    """ Increase the std of the measurement noise based on the distance to the landmark.
        The std will stay constant the first 10m.
        Then, it will linearly increase up to 3*std at MAX_RANGE.
    """
    std_out = std
    # Linear from 10 to MAX_RANGE
    if distance > 10:
        std_out = (2 * ((distance - 10) / (MAX_RANGE - 10)) + 1) * std
    # Linear addition with distance without limits
    # std_out = std + (0.01 * distance)
    return std_out


def compute_measurement(pose, landmark, std=MESAUREMENT_STD):
    """ Compute the measurement between pose_1 and pose_2.
        Add a gaussian noise in x and y coordinates with the given std.
        Inputs:
            - pose: SE(3) matrix containing the 6DOF position of the origin.
            - landmark: 3D position (x, y, z) of the landmark.
            - std: Standard deviation of the gaussian noise added in x and y axis.
        Output: A tuple, containing the measurement (a 2D array) and the covariance (2x2 matrix).
    """

    # Compute std depending on the distance to the landmark
    std = increase_range_error(std, distance(pose, landmark))
    # Build measurement covariance matrix
    measurement_cov = np.eye(2) * std**2

    # Convert the landmark pose to a SE(3) matrix
    landmark_se3 = lie.se3(t=landmark)

    # Compute measurement
    measurement = lie.relative_se3(pose, landmark_se3)

    # Add gaussian noise
    if std > 0.0:
        measurement[:2, 3] += np.random.multivariate_normal(np.zeros(2), measurement_cov)

    return measurement, measurement_cov


def filter_landmarks(landmarks, pose, max_range=MAX_RANGE):
    """ Get the subset of landmarks that can be ranged from the given pose (visibility model) """
    filtered_landmarks = []
    for l in range(landmarks.shape[0]):
        if distance(pose, landmarks[l]) < max_range:
            filtered_landmarks.append(landmarks[l])
    return np.array(filtered_landmarks)


def landmark_detection(pose, landmarks, std=MESAUREMENT_STD):
    """ Compute a measurement from the origin pose to each of the landmarks.
        Inputs:
            - pose: SE(3) matrix containing the 6DOF position of the origin.
            - landmarks: numpy matrix containing the 3D position (x, y, z) of each landmark (Nx3).
            - std: Standard deviation of the gaussian noise added in x and y axis.
        Outputs (tuple):
            - Measurements: Array of SE(3) matrices.
            - Measurement covariances: Array of 2x2 matrices.
    """
    measurements = np.empty((landmarks.shape[0], 4, 4))
    measurement_covs = np.empty((landmarks.shape[0], 2, 2))

    for l in range(landmarks.shape[0]):
        measurements[l], measurement_covs[l] = compute_measurement(pose, landmarks[l], std)
    return measurements, measurement_covs


def main():
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



if __name__ == '__main__':
    main()
