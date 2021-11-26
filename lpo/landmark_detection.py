#!/usr/bin/env python3

""" TODO: docstring """

import numpy as np

import lie_algebra as lie

# TODO: Find a better way to setup this
# MESAUREMENT_STD = 0.05
MESAUREMENT_STD = 'model'

MAX_RANGE = 30.0
# MAX_RANGE = np.inf

POLE_RADIUS = 0.085


def distance(pose, landmark):
    """ Compute the euclidean distance between an SE(3) pose and a landmark (3D array) """
    diff = landmark - pose[:3, 3]
    return np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)

def distance_2d(pose, landmark):
    """ Compute the euclidean distance between a 2D pose and a 2D landmark (2D arrays) """
    diff = landmark - pose
    return np.sqrt(diff[0]**2 + diff[1]**2)

def angle(pose, landmark):
    """ Compute the angle between an SE(3) pose and a landmark (3D array) in the X/Y plane """
    diff = landmark - pose[:3, 3]
    return np.arctan2(diff[1], diff[0])


# Error model functions from LPO calibration
def distance_std_model(distance):
    coeffs = [
        0.0005479803695850574,
        0.0007002317135424681
    ]
    line_fit_f = np.poly1d(coeffs)
    return line_fit_f(distance)


def get_transformed_covariance(distance, angle):
    distance_std = distance_std_model(distance)
    angle_std = 0.0010694619844429979 # rad
    angle_std_scaled = distance * np.tan(angle_std)
    polar_cov = np.array([
        [distance_std**2, 0.0],
        [0.0, angle_std_scaled**2],
    ])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    R_inv = np.linalg.inv(R)
    cartesian_cov = R.dot(polar_cov).dot(R_inv)
    return cartesian_cov


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

    l_distance = distance(pose, landmark)
    if std == "model":
        # New method based on calibrated model
        l_angle = angle(pose, landmark)
        measurement_cov = get_transformed_covariance(l_distance, l_angle)
    else:
        # Old method
        # Compute std depending on the distance to the landmark
        std = increase_range_error(std, l_distance)
        # Build measurement covariance matrix
        measurement_cov = np.eye(2) * std**2


    # Convert the landmark pose to a SE(3) matrix
    landmark_se3 = lie.se3(t=landmark)

    # Compute measurement
    measurement = lie.relative_se3(pose, landmark_se3)

    # Add gaussian noise
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
    angle_trans = (angle - angle_start) % (2*np.pi)
    return angle_trans <= alpha

def filter_landmarks_occlusions(landmarks, pose, map_data=None, map_resolution=None, max_range=MAX_RANGE):
    """ Get the subset of landmarks that can be ranged from the given pose and that are not occluded (visibility model) """
    # Get all cells that create occlusions together (from map and landmarks)
    occlusion_cells = []
    # First add landmarks filtered by distance
    for l in range(landmarks.shape[0]):
        cell_distance = distance(pose, landmarks[l])
        if cell_distance < max_range:
            # Store distance (for sorting), landmark object and boolean to identify landmarks
            occlusion_cells.append((cell_distance, landmarks[l], True))
    # Then add map obstacles filtered by distance
    if map_data is not None:
        obstacle_cells = np.argwhere(map_data == 0)
        for obstacle_cell in obstacle_cells:
            obstacle_pose = np.array([
                obstacle_cell[0] * map_resolution,
                obstacle_cell[1] * map_resolution,
                0.0,
            ])
            cell_distance = distance(pose, obstacle_pose)
            if cell_distance < max_range:
                # Store distance (for sorting), obstacle cell object and boolean to identify landmarks
                occlusion_cells.append((cell_distance, obstacle_pose, False))

    # Sort all occlusion cells by distance
    occlusion_cells.sort(key=lambda a: a[0])

    filtered_landmarks = []
    blocked_angles = []
    for cell_distance, occlusion_cell, is_landmark in occlusion_cells:
        # Compute occlusion_cell angle
        cell_angle = np.arctan2(occlusion_cell[1] - pose[1, 3], occlusion_cell[0] - pose[0, 3]) % (2*np.pi)
        if is_landmark:
            # Check if landmark is blocked by a closer obstacle and skip it
            for angle_start, alpha in blocked_angles:
                if angle_between(cell_angle, angle_start, alpha):
                    break
            else:
                # Landmark angle is not blocked. Add landmark to final list
                filtered_landmarks.append(occlusion_cell)
                # Compute FOV of new landmark and add it to the block list
                alpha = 2 * np.arctan2(POLE_RADIUS, cell_distance)
                angle_start = (cell_angle - alpha/2) % (2*np.pi)
                blocked_angles.append((angle_start, alpha))
        else:
            # Compute FOV of cell and add it to the block list
            alpha = 2 * np.arctan2(map_resolution/2, cell_distance)
            angle_start = (cell_angle - alpha/2) % (2*np.pi)
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
    measurement, measurement_cov = compute_measurement(pose, landmark_pose, 'model')
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
        [5.0, 0.0, 0.0],
        [15.0, 15.2, 0.0],
        [5.0, -5.0, 0.0],
        [-5.0, -5.0, 0.0],
        [30.0, 1.0, 0.0],
        [10.0, 1.0, 0.0],
    ])
    pose = lie.se3(t=[1.0, 1.0, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
    print("origin:\n{}".format(pose))
    print("landmarks:\n{}".format(landmarks))
    filtered_landmarks = filter_landmarks_occlusions(landmarks, pose, max_range=MAX_RANGE)
    print("filtered landmarks:\n{}".format(filtered_landmarks))
    measurements, measurement_covs = landmark_detection(pose, filtered_landmarks)
    print("measurements:\n{}".format(measurements))
    print("measurement_covs:\n{}".format(measurement_covs))



if __name__ == '__main__':
    main()
