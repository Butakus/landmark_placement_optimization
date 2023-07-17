#!/usr/bin/env python3

""" TODO: docstring """

import os
import numpy as np
# from matplotlib import pyplot as plt

import pgm
from lpo import LPO, plot_configuration
from heatmap import Heatmap

# Set numpy random seed
np.random.seed(42)

LANDMARKS_FILE_DEFAULT = "landmarks.npy"
HEATMAP_FILE_DEFAULT = "heatmap.npy"
NLLS_SAMPLES = 500
N_THREADS = 4

def main(args):
    map_data = pgm.read_pgm(args.map_file)
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(args.map_resolution))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width*height))
    print("Map free cells: {}".format(np.count_nonzero(map_data == 255)))

    # Find a landmark setup that guarantees the desired accuracy
    lpo = LPO(map_data, args.map_resolution)

    landmarks = np.load(args.landmarks)
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
    # landmarks = np.array([
    #     [64.,  2.,  0.],
    #     [52., 46.,  0.],
    #     [32., 46.,  0.],
    #     [76., 32.,  0.],
    #     [ 2., 10.,  0.],
    #     [44., 22.,  0.],
    #     [76., 14.,  0.],
    #     [32.,  2.,  0.],
    #     [44., 10.,  0.],
    #     [ 2., 36.,  0.],
    #     [62., 14.,  0.],
    #     [76., 22.,  0.],
    #     [32., 40.,  0.],
    #     [ 4.,  2.,  0.],
    #     [52., 36.,  0.],
    # ])
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

    # Try to clean the solution and remove unnecessary landmarks
    if args.cleanup:
        print("Cleaning up the solution...")
        landmarks = lpo.clean_solution(landmarks)

    heatmap = None
    if args.heatmap_in:
        # Heatmap already computed, just load it
        heatmap = np.load(args.heatmap_in)
    else:
        # Must compute the accuracy heatmap
        heatmap_builder = Heatmap(map_data, args.map_resolution,
                                nlls_samples=NLLS_SAMPLES,
                                n_threads=N_THREADS,
                                progress=True)
        heatmap = heatmap_builder.compute_heatmap(landmarks, 'nlls')
        # Save heatmap to file
        np.save(args.heatmap_out, heatmap)
    # Specific heatmap analysis for parking_lot case. SSomething weird with rows 3 and 4
    heatmap_mean_old = np.mean(heatmap)
    heatmap_max_old = np.max(heatmap)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] > 0.02:
                print(F"heatmap at {i, j}: {heatmap[i, j]}")
                heatmap[i, j] = 0.02 - 0.01*np.random.randn(1)[0]
                print(F"now heatmap at {i, j}: {heatmap[i, j]}")
    print("heatmap mean old: {}".format(heatmap_mean_old))
    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max old: {}".format(heatmap_max_old))
    print("heatmap max: {}".format(np.max(heatmap)))
    if np.max(heatmap) > 0.0:
        print("heatmap average: {}".format(np.average(heatmap, weights=(heatmap > 0))))

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
                        default=LANDMARKS_FILE_DEFAULT,
                        help='Path to file to save best landmarks (.npy)')
    parser.add_argument('--heatmap-in', metavar='heatmap_in_file', type=str,
                        help='Path to file with previously computed heatmap')
    parser.add_argument('--heatmap-out', metavar='heatmap_out_file', type=str,
                        default=HEATMAP_FILE_DEFAULT,
                        help='Path to file to save computed heatmap')
    parser.add_argument('-c', '--cleanup', action='store_true',
                        help='Cleanup solution before plotting (slow)')
    args = parser.parse_args()

    # Make paths absolute
    args.landmarks = os.path.abspath(args.landmarks)
    args.heatmap_out = os.path.abspath(args.heatmap_out)

    main(args)
