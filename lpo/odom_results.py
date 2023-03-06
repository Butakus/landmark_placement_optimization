#!/usr/bin/env python3

""" TODO: docstring """

import os
from tqdm import tqdm
import csv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import pgm
import lie_algebra as lie
from landmark_detection import landmark_detection, filter_landmarks, filter_landmarks_occlusions
import nlls

# Set numpy random seed
np.random.seed(42)

RESOLUTION = 1
# MAP_FILE = "/home/butakus/localization_reference/gazebo/parque_maps/map_{r}p0.pgm".format(r=RESOLUTION)
MAP_FILE = "/home/butakus/localization_reference/gazebo/map_1_{r}p0.pgm".format(r=RESOLUTION)
RESOLUTION = float(RESOLUTION)
# ODOM_FILES = [
#     "/home/butakus/localization_reference/bagfiles/parque/run0_odom_filt.csv",
#     "/home/butakus/localization_reference/bagfiles/parque/run1_odom_filt.csv",
#     "/home/butakus/localization_reference/bagfiles/parque/run2_odom_filt.csv",
# ]
ODOM_FILES = [
    "/home/butakus/localization_reference/bagfiles/scenario_2/scenario_2_run_0.csv",
    "/home/butakus/localization_reference/bagfiles/scenario_2/scenario_2_run_1.csv",
    "/home/butakus/localization_reference/bagfiles/scenario_2/scenario_2_run_2.csv",
]
NLLS_ODOM_FILES = [
    "scenario_2_run_0_nlls_odom.npy",
    "scenario_2_run_1_nlls_odom.npy",
    "scenario_2_run_2_nlls_odom.npy",
]
DROP = 2

RUN_NLLS = False

NLLS_SAMPLES = 10
TARGET_ACCURACY = 0.015

# landmarks = np.load(
#     "/home/butakus/localization_reference/landmark_placement_optimization/logs/landmarks_5p0_1.npy"
# )
# landmarks = np.load(
#     "/home/butakus/localization_reference/landmark_placement_optimization/logs/landmarks_5p0_0p05.npy"
# )
# Cheating to make display pretty (because landmarks were computed for 5m resolution and display is 1m)
# for i in range(landmarks.shape[0]):
#     if landmarks[i, 0] == 10.0:
#         landmarks[i, 0] = 9.0
#     if landmarks[i, 0] == 100.0 and landmarks[i, 1] == 45.0:
#         landmarks[i, 1] = 37.0
#     if landmarks[i, 0] == 95.0 and landmarks[i, 1] == 45.0:
#         landmarks[i, 0] = 96.0
#         landmarks[i, 1] = 37.0
# landmarks = np.array([
#     [5.0, 40.0, 0.0],
#     [40.0, 50.0, 0.0],
#     [55.0, 25.0, 0.0],
#     [65.0, 55.0, 0.0],
#     [85.0, 60.0, 0.0],
#     [100.0, 30.0, 0.0],
#     [120.0, 60.0, 0.0],
#     [125.0, 23.0, 0.0],
# ])
# landmarks = np.array([
#     [4.0, 40.0, 0.0],
#     [40.0, 50.0, 0.0],
#     [54.0, 24.0, 0.0],
#     [64.0, 54.0, 0.0],
#     [68.0, 30.0, 0.0],
#     [124.0, 22.0, 0.0],
# ])
# landmarks = np.array([
#     [5.0, 40.0, 0.0],
#     [40.0, 45.0, 0.0],
#     [55.0, 25.0, 0.0],
#     [65.0, 55.0, 0.0],
#     [125.0, 20.0, 0.0],
# ])


"""
DROP=2 & NLLS_SAMPLES=5
##########################
Odom run 0
Trans error mean:   0.002305866326346101
Trans error std:    0.0014079795689448942
Max Trans error:    0.013547463672799617
Rot error mean: 0.000948191555152663
Rot error std:  0.0008376196786161676
Max rot error:  0.003539928945444259

##########################
Odom run 1
Trans error mean:   0.00249141949654804
Trans error std:    0.0017524373013097585
Max Trans error:    0.014549625979962622
Rot error mean: 0.0012624291374199405
Rot error std:  0.0009463382925669817
Max rot error:  0.004339026467921374

##########################
Odom run 2
Trans error mean:   0.0028958762178741093
Trans error std:    0.0018546364375405825
Max Trans error:    0.014998836719466509
Rot error mean: 0.005833792257481558
Rot error std:  0.006374654710008905
Max rot error:  0.041178056394115816

"""


# scenario 2 1p0
landmarks = np.array([
    [64.,  3.,  0.],
    [52., 46.,  0.],
    [32., 46.,  0.],
    [76., 32.,  0.],
    [ 3., 10.,  0.],
    [44., 23.,  0.],
    [76., 14.,  0.],
    [32.,  3.,  0.],
    [44., 10.,  0.],
    [ 3., 36.,  0.],
    [62., 14.,  0.],
    [76., 24.,  0.],
    [32., 40.,  0.],
    [ 4.,  3.,  0.],
    [52., 36.,  0.],
])


print(F"Landmarks: {landmarks}")


def read_odom(odom_file, drop=0):
    odom_x = []
    odom_y = []
    odom_theta = []
    with open(odom_file, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',')
        drop_count = 0
        for row in csv_reader:
            if drop_count == drop:
                drop_count = 0
                odom_x.append(float(row['pose.position.x']))
                odom_y.append(float(row['pose.position.y']))
                # Compute theta from quaternion
                quat = [
                    float(row['pose.orientation.w']),
                    float(row['pose.orientation.x']),
                    float(row['pose.orientation.y']),
                    float(row['pose.orientation.z']),
                ]
                (yaw, pitch, roll) = lie.euler_from_quaternion(quat)
                # odom_theta.append(yaw + np.pi)
                # odom_theta.append(np.pi/3)
                odom_theta.append(np.pi/2)
            else:
                drop_count += 1
    # print(F"Theta: {odom_theta}")
    # print(F"mean Theta: {np.mean(odom_theta)}")
    # print(F"std Theta: {np.std(odom_theta)}")
    # print(F"min Theta: {np.min(odom_theta)}")
    # print(F"max Theta: {np.max(odom_theta)}")
    # plt.hist(odom_theta)
    # plt.show()
    odom = np.array([odom_x, odom_y, odom_theta]).T
    print(odom.shape)
    return odom


def compute_nlls_odom(odom, map_data=None, map_resolution=None):
    nlls_odom = []
    for i in tqdm(range(odom.shape[0])):
        odom_se3 = lie.se3(t=[odom[i, 0], odom[i, 1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, odom[i, 2]]))
        results = []
        for _ in range(NLLS_SAMPLES):
            # Initialize initial guess
            initial_guess = odom[i].copy()
            # Add noise to initial solution
            initial_guess[:2] += np.random.normal(0.0, 0.5, 2)
            initial_guess[2] += np.random.normal(0.0, 0.2)
            # Get the landmark measurements
            # filtered_landmarks = filter_landmarks(landmarks, odom_se3)
            filtered_landmarks = filter_landmarks_occlusions(landmarks, odom_se3, map_data=map_data, map_resolution=map_resolution)
            measurements, measurement_covs = landmark_detection(odom_se3, filtered_landmarks)
            # Estimate the NLLS solution
            nlls_result = nlls.nlls_estimation(args=(filtered_landmarks, measurements, measurement_covs),
                                               initial_guess=None, output=False)
            results.append(nlls_result)
        x = np.array([r.params['x'] for r in results])
        y = np.array([r.params['y'] for r in results])
        theta = np.array([r.params['theta'] for r in results])

        std_x = np.std(x)
        std_y = np.std(y)
        # Filter outlier estimations (error > 3*std)
        filt_x = []
        filt_y = []
        eps = 100 * np.finfo(type(std_x)).eps
        for n in range(NLLS_SAMPLES):
            dx = abs(x[n] - odom[i, 0])
            dy = abs(y[n] - odom[i, 1])
            if (std_x < eps or dx <= 3*std_x) and (std_y < eps or dy <= 3*std_y):
                filt_x.append(x[n])
                filt_y.append(y[n])
            else:
                print("Filter out estimation point: {} / {}".format(x[n], y[n]))
                print("Odom: {} / {}".format(odom[i, 0], odom[i, 1]))
                print("dx/dy: {} / {}".format(dx, dy))
                print("std: {} / {}".format(std_x, std_y))
        filt_x = np.array(filt_x)
        filt_y = np.array(filt_y)

        if filt_x.shape[0] == 0:
            # We removed all points, this is crazy. Use all of them.
            print("OUT OF POINTS")
            filt_x = x
            filt_y = y

        nlls_odom.append([
            np.mean(filt_x),
            np.mean(filt_y),
            np.mean(theta),
            # np.mean(theta) + np.random.normal(0.0, np.deg2rad(0.05)), # I don't know why I added this
        ])

    print(np.array(nlls_odom))
    return np.array(nlls_odom)


def get_odom_error(odom, nlls_odom):
    trans_error = []
    rot_error = []
    failure_count = 0
    for i in range(odom.shape[0]):
        # print("---------------------------")
        # print(F"odom: {odom[i, :2]}")
        # print(F"nlls: {nlls_odom[i, :2]}")
        dx = nlls_odom[i, 0] - odom[i, 0]
        dy = nlls_odom[i, 1] - odom[i, 1]
        dtheta = nlls_odom[i, 2] - odom[i, 2]
        e = np.sqrt(dx**2 + dy**2)

        # Check if error is higher than 3*max_std
        # This is 3 times the target accuracy. 99% of the data should be below this threshold.
        if e > 3*TARGET_ACCURACY:
            print("FAIL -> {}".format(e))
            failure_count += 1
            if e > 6*TARGET_ACCURACY:
                print("SUPER FAIL -> {}".format(e))
                # print("Dropping measurement")
                # continue
        trans_error.append(e)
        rot_error.append(abs(np.rad2deg(dtheta)))
    print(F"Number of failures: {failure_count}. Number of poses: {odom.shape[0]}. Failure rate: {100*failure_count/odom.shape[0]}%")
    return np.array(trans_error), np.array(rot_error)


def main():
    map_data = pgm.read_pgm(MAP_FILE)
    width, height = map_data.shape
    # print(map_data)
    print("resolution: {}".format(RESOLUTION))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width*height))
    print("Map free cells: {}".format(np.count_nonzero(map_data)))

    odoms = []
    odoms_display = []
    nlls_odoms_display = []
    odoms_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green"
    ]
    nlls_odoms = []
    odom_trans_error = []
    odom_rot_error = []
    for i, odom_file in enumerate(ODOM_FILES):
        odom = read_odom(odom_file, drop=DROP)
        odoms.append(odom)
        odoms_display.append(odom / RESOLUTION)
        if RUN_NLLS:
            # file_name = os.path.basename(odom_file).rstrip('.csv').replace('odom_filt', 'nlls_odom')
            file_name = os.path.basename(odom_file).replace('.csv', '_nlls_odom')
            nlls_odom = compute_nlls_odom(odom, map_data=map_data, map_resolution=RESOLUTION)
            np.save(file_name, nlls_odom)
        else:
            nlls_odom = np.load(NLLS_ODOM_FILES[i])
            print(nlls_odom)
        nlls_odoms.append(nlls_odom)
        nlls_odoms_display.append(nlls_odom / RESOLUTION)
        trans_error, rot_error = get_odom_error(odom, nlls_odom)
        print(F"trans_error:\n{trans_error}")
        print(F"rot_error:\n{rot_error}")
        odom_trans_error.append(trans_error)
        odom_rot_error.append(rot_error)

    # exit()

    for i, odom_file in enumerate(ODOM_FILES):
        print("##########################")
        print(F"Odom run {i}")
        print(F"Trans error mean:\t{np.mean(odom_trans_error[i])}")
        print(F"Trans error std:\t{np.std(odom_trans_error[i])}")
        print(F"Max Trans error:\t{np.max(odom_trans_error[i])}")
        print(F"Rot error mean:\t{np.mean(odom_rot_error[i])}")
        print(F"Rot error std:\t{np.std(odom_rot_error[i])}")
        print(F"Max rot error:\t{np.max(odom_rot_error[i])}")

    # Display the maps for the obtained landmark configuration
    map_display = map_data.transpose()
    landmarks_display = landmarks / RESOLUTION
    plt.imshow(map_display, cmap='gray', origin='lower')
    plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m', s=200)
    triangle = mlines.Line2D([], [], color='m', marker='^', linestyle='None', markersize=10, label='Landmarks')
    legend_handles = [triangle]
    for i, odom_d in enumerate(odoms_display):
        plt.scatter(odom_d[:, 0], odom_d[:, 1],
                    color=odoms_colors[i], marker=',', s=1)
        legend_handles.append(mpatches.Patch(color=odoms_colors[i], label='Trajecotry {}'.format(i+1)))
        # plt.scatter(nlls_odoms_display[i][:, 0], nlls_odoms_display[i][:, 1], color='r', marker='.', s=1)

    # plt.legend()
    plt.legend(handles=legend_handles)
    # Plot error
    plt.figure()
    plt.plot(odom_trans_error[0])
    plt.figure()
    plt.plot(odom_rot_error[0])
    # Plot error CDF
    plt.figure()
    plt.xlabel('Translation error (m)')
    plt.ylabel('Probability')
    for i, odom_error in enumerate(odom_trans_error):
        sorted_error = np.sort(odom_error)
        cdf = np.arange(1, len(sorted_error)+1)/float(len(sorted_error))
        # plt.hist(odom_error, density=True, cumulative=True, label="CDF error", histtype='step', bins=100)
        plt.title('Translation error CDF')
        plt.plot(sorted_error, cdf, label=F"Odom run {i}")
    # Draw vert lines for std and 3*std
    plt.axvline(x=TARGET_ACCURACY, ls='--', color='y', ymin=0.05, ymax=0.98, label="Target std")
    plt.axvline(x=3*TARGET_ACCURACY, ls='--', color='r', ymin=0.05, ymax=0.98, label="3 * Target std")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
