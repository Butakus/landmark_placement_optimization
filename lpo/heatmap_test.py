#!/usr/bin/env python3

""" TODO: docstring """

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

import pgm
import metrics
import lie_algebra as lie
from landmark_detection import landmark_detection
from heatmap import Heatmap

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
#     [ 45.,  45.,   0.,],
#     [ 85.,  55.,   0.,],
#     [ 75.,  55.,   0.,],
#     [105.,  25.,   0.,],
#     [ 60.,  30.,   0.,],
#     [140.,  55.,   0.,],
#     [ 65.,  45.,   0.,],
# ])
landmarks = np.array([
    [90., 25., 0.],
    [50., 40., 0.],
    [65., 25., 0.],
    [90., 30., 0.],
    [5., 50., 0.],
    [145., 50., 0.],
    [60., 40., 0.]
])


def map_gdop(map_data, map_resolution):
    gdop_map = np.zeros(map_data.shape)
    for i in tqdm(range(map_data.shape[0])):
        for j in range(map_data.shape[1]):
            # Only map the area if it is drivable
            if map_data[i, j] == 255:
                cell_pose = lie.se3(t=[i * map_resolution, j * map_resolution, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
                measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.001)
                try:
                    gdop_map[i][j] = metrics.compute_gdop(measurements)
                    if gdop_map[i][j] > 250.0:  # or np.isnan(gdop_map[i][j]):
                        # print("WWWW!!!")
                        # print("Cell: {}".format(cell_pose))
                        # print("gdop: {}".format(gdop_map[i][j]))
                        gdop_map[i][j] = 250.0
                except np.linalg.LinAlgError:
                    gdop_map[i][j] = 250.0
    return gdop_map


def test_gdop(x, y):
    X = np.array([x, y, 0.0])
    print("X: {}".format(X))

    # Get the landmark measurements
    X_se3 = lie.se3(t=[X[0], X[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, X[2]]))
    measurements, measurement_covs = landmark_detection(X_se3, landmarks, std=0.0)
    cell_gdop = metrics.compute_gdop(measurements)
    return cell_gdop


def main(args):
    map_data = pgm.read_pgm(args.map_file)
    resolution = args.map_resolution
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(resolution))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width * height))
    print("Map free cells: {}".format(np.count_nonzero(map_data)))

    # Transpose the map data matrix because imshow will treat it
    # as an image and use the first dimension as the Y axis (rows)
    # This is needed only for displaying the map image
    map_display = map_data.transpose()
    # Cnvert landmarks from meters to pixel coordinates
    landmarks_display = landmarks / resolution
    plt.imshow(map_display, cmap='gray', origin='lower')
    plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')

    # Compute GDOP:
    # gdop_map = map_gdop(map_data)
    # print(gdop_map)
    # print("gdop mean: {}".format(np.mean(gdop_map)))
    # print("gdop max: {}".format(np.max(gdop_map)))

    # # Create a mask to dislpay the DGOP on top of the map and transpose for displaying
    # gdop_map_masked = np.ma.masked_where(gdop_map == 0, gdop_map).transpose()
    # plt.imshow(gdop_map_masked, 'cubehelix', interpolation='none', alpha=0.7)

    # plt.figure()
    # plt.hist(gdop_map)
    # plt.figure()
    # plt.imshow(gdop_map, cmap='viridis', origin='lower')
    # plt.show()
    # exit()

    # Compute NLLS posterior:
    heatmap_gen = Heatmap(map_data, resolution)
    heatmap = heatmap_gen.compute_heatmap(landmarks, metric='mcmc')

    # print(heatmap)

    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max: {}".format(np.max(heatmap)))
    if np.max(heatmap) > 0.0:
        print("heatmap average: {}".format(np.average(heatmap, weights=(heatmap > 0))))

    # Create a mask to dislpay the heatmap on top of the map and transpose for displaying
    heatmap_masked = np.ma.masked_where(map_data == 0, heatmap).transpose()
    plt.imshow(heatmap_masked, 'viridis', interpolation='none', alpha=1.0, origin='lower')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Heatmap module test')
    parser.add_argument('map_file', metavar='map_file', type=str,
                        help='Map pgm file')
    parser.add_argument('map_resolution', metavar='map_resolution', type=float,
                        help='Map resolution (m/cell)')
    parser.add_argument('-l', '--landmarks', metavar='landmarks_file', type=str,
                        help='Path to landmarks file (.npy)')
    args = parser.parse_args()

    # Use landmark positions from file
    if args.landmarks:
        landmarks = np.load(args.landmarks)

    main(args)
