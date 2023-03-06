#!/usr/bin/env python3

""" TODO: docstring """

import os
import numpy as np
# from matplotlib import pyplot as plt

import pgm
from lpo import LPO, plot_configuration

# Set numpy random seed
np.random.seed(42)

LANDMARKS_FILE = "landmarks.npy"


def main(args):
    map_data = pgm.read_pgm(args.map_file)
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(args.map_resolution))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width*height))
    print("Map free cells: {}".format(np.count_nonzero(map_data)))

    # Find a landmark setup that guarantees the desired accuracy
    lpo = LPO(map_data, args.map_resolution)

    # landmarks = np.load(LANDMARKS_FILE)
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

    # 2p0
    landmarks = np.array([
        [64.,  2.,  0.],
        [52., 46.,  0.],
        [32., 46.,  0.],
        [76., 32.,  0.],
        [ 2., 10.,  0.],
        [44., 22.,  0.],
        [76., 14.,  0.],
        [32.,  2.,  0.],
        [44., 10.,  0.],
        [ 2., 36.,  0.],
        [62., 14.,  0.],
        [76., 22.,  0.],
        [32., 40.,  0.],
        [ 4.,  2.,  0.],
        [52., 36.,  0.],
    ])
    # 1p0
    # landmarks = np.array([
    #     [64.,  3.,  0.],
    #     [52., 46.,  0.],
    #     [32., 46.,  0.],
    #     [76., 32.,  0.],
    #     [ 3., 10.,  0.],
    #     [44., 23.,  0.],
    #     [76., 14.,  0.],
    #     [32.,  3.,  0.],
    #     [44., 10.,  0.],
    #     [ 3., 36.,  0.],
    #     [62., 14.,  0.],
    #     [76., 24.,  0.],
    #     [32., 40.,  0.],
    #     [ 4.,  3.,  0.],
    #     [52., 36.,  0.],
    # ])
    # landmarks = np.load(LANDMARKS_FILE)


    # Coverage
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

    # lpo.find_cell_max_coverage(landmarks=np.empty((0, 3)))
    # lpo.find_cell_max_coverage(landmarks=landmarks)

    # lpo.find_cell_max_coverage_2(landmarks=np.empty((0, 3)))
    # lpo.find_cell_max_coverage_2(landmarks=landmarks)
    # exit()

    print("landmarks:\n{}".format(landmarks))

    valid = lpo.valid_configuration(landmarks)
    print("Valid configuration: {}".format(valid))

    heatmap = lpo.heatmap_builder.compute_heatmap(landmarks, 'nlls')
    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max: {}".format(np.max(heatmap)))
    if np.max(heatmap) > 0.0:
        print("heatmap average: {}".format(np.average(heatmap, weights=(heatmap > 0))))
    # heatmap = None

    coverage = lpo.get_coverage_map(landmarks)
    print(coverage)
    print("coverage mean: {}".format(np.mean(coverage)))
    print("coverage max: {}".format(np.max(coverage)))
    if np.max(coverage) > 0.0:
        print("coverage average: {}".format(np.average(coverage, weights=(coverage > 0))))
    print("coverage fitness: {}".format(lpo.fair_coverage_fitness(coverage)))

    # Display the maps for the obtained landmark configuration
    plot_configuration(map_data, args.map_resolution, landmarks, heatmap=heatmap, coverage=coverage)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot tool to visualize LPO results')
    parser.add_argument('map_file', metavar='map_file', type=str,
                        help='Map pgm file')
    parser.add_argument('map_resolution', metavar='map_resolution', type=float,
                        help='Map resolution (m/cell)')
    parser.add_argument('-l', '--landmarks', metavar='landmarks_file', type=str,
                        help='Path to file to save best landmarks (.npy)')
    args = parser.parse_args()

    # Change landmarks file if needed and make path absolute
    if args.landmarks and not os.path.isdir(args.landmarks):
        LANDMARKS_FILE = args.landmarks
    LANDMARKS_FILE = os.path.abspath(LANDMARKS_FILE)

    main(args)
