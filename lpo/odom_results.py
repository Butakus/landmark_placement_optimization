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

LANDMARKS_FILE_DEFAULT = "landmarks.npy"
DROP_DEFAULT = 2

NLLS_SAMPLES = 10
TARGET_STD_DEFAULT = 0.01

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


def compute_nlls_odom(odom, landmarks, map_data=None, map_resolution=None):
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

    print("Odom:\n{}".format(np.array(nlls_odom)))
    return np.array(nlls_odom)


def get_odom_error(odom, nlls_odom, target_std=TARGET_STD_DEFAULT):
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
        if e > 3*target_std:
            print("FAIL -> {}".format(e))
            failure_count += 1
            if e > 6*target_std:
                print("SUPER FAIL -> {}".format(e))
                # print("Dropping measurement")
                # continue
        trans_error.append(e)
        rot_error.append(abs(np.rad2deg(dtheta)))
    print(F"Number of failures: {failure_count}. Number of poses: {odom.shape[0]}. Failure rate: {100*failure_count/odom.shape[0]}%")
    return np.array(trans_error), np.array(rot_error)


def main(args):
    map_data = pgm.read_pgm(args.map_file)
    width, height = map_data.shape
    landmarks = np.load(args.landmarks)
    # print(map_data)
    print(F"resolution: {args.map_resolution}")
    print(F"width: {width}")
    print(F"height: {height}")
    print(F"Map cells: {width*height}")
    print(F"Map free cells: {np.count_nonzero(map_data == 255)}")
    print(F"Landmarks:\n{landmarks}")

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

    # Check if we can skip NLLS computation
    odom_files = args.odom
    nlls_files = None
    run_nlls = True
    if args.nlls:
        nlls_files = args.nlls
        run_nlls = False
        if len(nlls_files) != len(odom_files):
            print("Error: Length of odom and nlls files mismatch")
            exit(1)

    print(F"odom_files: {odom_files}")
    print(F"nlls_files: {nlls_files}")
    print(F"run_nlls: {run_nlls}")

    for i, odom_file in enumerate(odom_files):
        odom = read_odom(odom_file, drop=args.drop)
        odoms.append(odom)
        odoms_display.append(odom / args.map_resolution)
        if run_nlls:
            # file_name = os.path.basename(odom_file).rstrip('.csv').replace('odom_filt', 'nlls_odom')
            file_dir = os.path.dirname(odom_file)
            file_name = os.path.basename(odom_file).replace('.csv', '_nlls_odom')
            file_path = os.path.join(file_dir, file_name)
            nlls_odom = compute_nlls_odom(odom, landmarks, map_data=map_data, map_resolution=args.map_resolution)
            print(F"Saving nlls odom file to {file_path}")
            np.save(file_path, nlls_odom)
        else:
            nlls_odom = np.load(nlls_files[i])
            print(nlls_odom)
        nlls_odoms.append(nlls_odom)
        nlls_odoms_display.append(nlls_odom / args.map_resolution)
        trans_error, rot_error = get_odom_error(odom, nlls_odom, target_std=args.target_std)
        # print(F"trans_error:\n{trans_error}")
        # print(F"rot_error:\n{rot_error}")
        odom_trans_error.append(trans_error)
        odom_rot_error.append(rot_error)

    for i, odom_file in enumerate(odom_files):
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
    landmarks_display = landmarks / args.map_resolution
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
    plt.axvline(x=args.target_std, ls='--', color='y', ymin=0.05, ymax=0.98, label="Target std")
    # plt.axvline(x=3*args.target_std, ls='--', color='r', ymin=0.05, ymax=0.98, label="3 * Target std")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tool to compare the estimated odometry with the ground truth')
    parser.add_argument('map_file', metavar='map_file', type=str,
                        help='Map pgm file')
    parser.add_argument('map_resolution', metavar='map_resolution', type=float,
                        help='Map resolution (m/cell)')
    parser.add_argument('-l', '--landmarks', metavar='landmarks_file', type=str,
                        default=LANDMARKS_FILE_DEFAULT,
                        help='Path to file to save best landmarks (.npy)')
    parser.add_argument('--odom', metavar='odom_files', nargs='+', required=True,
                        help='Odometry ground truth files')
    parser.add_argument('--nlls', metavar='nlls_files', nargs='+',
                        help='NLLS localization files. Set this to avoid rerunning the NLLS algorithm.')
    parser.add_argument('-d', '--drop', metavar='drop', type=int,
                        default=DROP_DEFAULT,
                        help='Read only the nth odometry measurement from the ground truth to skip frames and make it faster')
    parser.add_argument('--target-std', type=float, default=TARGET_STD_DEFAULT,
                        help='Target std used in this runs. Used for plotting.')
    args = parser.parse_args()

    # Make paths absolute
    args.landmarks = os.path.abspath(args.landmarks)

    main(args)
