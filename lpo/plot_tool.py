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
