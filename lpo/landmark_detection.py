#!/usr/bin/env python3

""" TODO: docstring """

import numpy as np

import lie_algebra as lie

# TODO: Find a better way to setup this
MESAUREMENT_STD = 0.05

MAX_RANGE = 60.0
# MAX_RANGE = np.inf

POLE_RADIUS = 0.085


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

def angle_between(angle, angle_start, alpha):
    """ Check if a given angle is in the range between [angle_start, angle_start + alpha] """
    print("angle_between {s} - {a}".format(s=angle_start, a=alpha))
    angle_trans = (angle - angle_start) % (2*np.pi)
    print(angle_trans <= alpha)
    return angle_trans <= alpha

def filter_landmarks_occlusions(landmarks, pose, max_range=MAX_RANGE):
    """ Get the subset of landmarks that can be ranged from the given pose and that are not occluded (visibility model) """
    sorted_landmarks = []
    # First filter and sort by distance
    for l in range(landmarks.shape[0]):
        l_distance = distance(pose, landmarks[l])
        if l_distance < max_range:
            sorted_landmarks.append((l_distance, landmarks[l]))
    sorted_landmarks.sort(key=lambda a: a[0])
    print("sorted_landmarks:\n{}".format(sorted_landmarks))
    filtered_landmarks = []
    blocked_angles = []
    for l_distance, landmark in sorted_landmarks:
        # Comppute landmark angle
        print("-----------------------")
        print("blocked_angles:\n{}".format(blocked_angles))
        print("landmark:\n{}".format(landmark))
        landmark_angle = np.arctan2(landmark[1], landmark[0]) % (2*np.pi)
        print("landmark_angle:\n{}".format(landmark_angle))
        # Check if landmark is blocked by a closer landmark and skip it
        blocked_angle = False
        for angle_start, alpha in blocked_angles:
            blocked_angle = angle_between(landmark_angle, angle_start, alpha)
            if blocked_angle:
                break
        if blocked_angle:
            continue
        # Add landmark to final list
        filtered_landmarks.append(landmark)
        # Compute FOV of new landmark and add it to the block list
        alpha = 2 * np.arctan2(POLE_RADIUS, l_distance)
        angle_start = (landmark_angle - alpha/2) % (2*np.pi)
        blocked_angles.append((angle_start, alpha))

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
    # landmarks = np.array([
    #     [-5.0, 5.0, 0.0],
    #     [5.0, 5.0, 0.0],
    #     [5.0, -5.0, 0.0],
    #     [-5.0, -5.0, 0.0],
    # ])
    landmarks = np.array([
        [-5.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [15.0, 15.2, 0.0],
        [5.0, -5.0, 0.0],
        [-5.0, -5.0, 0.0],
        [30.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    pose = lie.se3(t=[1.0, 1.0, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
    print("origin: {}".format(pose))
    print("landmarks:\n{}".format(landmarks))
    filtered_landmarks = filter_landmarks_occlusions(landmarks, pose, max_range=MAX_RANGE)
    print("filtered landmarks:\n{}".format(filtered_landmarks))
    measurements, measurement_covs = landmark_detection(pose, filtered_landmarks)
    print("measurements:\n{}".format(measurements))
    print("measurement_covs:\n{}".format(measurement_covs))



if __name__ == '__main__':
    main()
