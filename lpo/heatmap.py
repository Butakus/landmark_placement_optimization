#!/usr/bin/env python3

""" TODO: docstring """

from time import time
import multiprocessing as mp

import numpy as np
# Set numpy random seed
# np.random.seed(42)
from matplotlib import pyplot as plt

from tqdm import tqdm

import pgm
import metrics
import nlls
import lie_algebra as lie
from landmark_detection import landmark_detection, filter_landmarks


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
# landmarks = np.load("landmarks.npy")

map_file = "/home/butakus/localization_reference/gazebo/map_5p0.pgm"
resolution = 5.0


def map_gdop(map_data):
    gdop_map = np.zeros(map_data.shape)
    for i in tqdm(range(map_data.shape[0])):
        for j in range(map_data.shape[1]):
            # Only map the area if it is drivable
            if map_data[i, j] == 255:
                cell_pose = lie.se3(t=[i * resolution, j * resolution, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
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


class Heatmap(object):
    """ TODO: docstring for Heatmap """

    def __init__(self, map_data, resolution):
        super(Heatmap, self).__init__()
        self.map_data = map_data
        self.resolution = resolution
        self.heatmap = np.zeros(self.map_data.shape)
        self.metrics = {
            'nlls': self.nlls_metric,
            'mcmc': self.mcmc_metric,
        }
        self.results = []
        self.landmarks = landmarks

    def add_async_result(self, result):
        i, j, val = result
        self.heatmap[i, j] = val

    def compute_heatmap(self, landmarks, metric):
        self.landmarks = landmarks
        metric_f = self.metrics[metric]
        # pool = mp.Pool(mp.cpu_count())
        # pool = mp.Pool(12)
        t0 = time()
        with mp.get_context("spawn").Pool(12) as pool:
            for i in range(self.map_data.shape[0]):
                for j in range(self.map_data.shape[1]):
                    # Only map the area if it is drivable
                    if self.map_data[i, j] == 255:
                        pool.apply_async(metric_f, args=(i, j), callback=self.add_async_result)

            print("Computing heatmap...")
            pool.close()
            pool.join()

        t1 = time()
        print("Heatmap build time: {}".format(t1 - t0))

        return self.heatmap

    def nlls_metric(self, i, j):
        """ TODO: docstring """
        cell_t = np.array([i * self.resolution, j * self.resolution, np.pi/2])
        cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, cell_t[2]]))
        # Get the subset of landmarks that are in range from the current cell
        filtered_landmarks = filter_landmarks(self.landmarks, cell_pose)
        if filtered_landmarks.shape[0] < 3:
            # Insufficient number of landmarks in range to solve the NLLS problem
            print("WARNING: Not enough landmarks in range!! Cell: {}".format(cell_t))
            return (i, j, np.inf)

        N = 10  # Take N samples to make it more robust

        x = np.zeros(N)
        y = np.zeros(N)

        for n in range(N):
            measurements, measurement_covs = landmark_detection(cell_pose, filtered_landmarks, std=0.05)
            initial_guess = cell_t.copy()
            initial_guess[:2] += np.random.normal(0.0, 1.0, 2)
            initial_guess[2] += np.random.normal(0.0, 0.2)
            try:
                nlls_result = nlls.nlls_estimation(args=(filtered_landmarks, measurements, measurement_covs),
                                                   initial_guess=initial_guess, output=False)
            except Exception as e:
                print("NLLS EXCEPTION: {}".format(e))
                return (i, j, np.inf)
            x[n] = nlls_result.params['x']
            y[n] = nlls_result.params['y']

        std_x = np.std(x)
        std_y = np.std(y)

        filt_x = []
        filt_y = []
        eps = 100 * np.finfo(type(std_x)).eps
        for n in range(N):
            dx = abs(x[n] - cell_t[0])
            dy = abs(y[n] - cell_t[1])
            if (std_x < eps or dx <= 3*std_x) and (std_y < eps or dy <= 3*std_y):
                filt_x.append(x[n])
                filt_y.append(y[n])
        if len(filt_x) != 0:
            filt_x = np.array(filt_x)
            filt_y = np.array(filt_y)

            std_x = np.std(filt_x)
            std_y = np.std(filt_y)

        error = np.sqrt(std_x**2 + std_y**2)

        return (i, j, error)

    def mcmc_metric(self, i, j):
        cell_t = np.array([i * self.resolution, j * self.resolution, np.pi / 2])
        cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, cell_t[2]]))
        # Get the subset of landmarks that are in range from the current cell
        filtered_landmarks = filter_landmarks(self.landmarks, cell_pose)
        if filtered_landmarks.shape[0] < 3:
            # Insufficient number of landmarks in range to solve the NLLS problem
            print("WARNING: Not enough landmarks in range!! Cell: {}".format(cell_t))
            return (i, j, np.inf)

        measurements, measurement_covs = landmark_detection(cell_pose, filtered_landmarks, std=0.01)

        initial_guess = cell_t.copy()
        initial_guess[:2] += np.random.normal(0.0, 1.0, 2)
        initial_guess[2] += np.random.normal(0.0, 0.2)

        # Sometimes problem here. It is raising exception or something
        nlls_result = nlls.nlls_estimation(args=(filtered_landmarks, measurements, measurement_covs),
                                           initial_guess=initial_guess, output=False)
        emcee_result = nlls.mcmc_posterior_estimation(params=nlls_result.params, steps=3000,
                                                      args=(filtered_landmarks, measurements, measurement_covs),
                                                      output=False)
        std_x = emcee_result.params['x'].stderr
        std_y = emcee_result.params['y'].stderr
        return (i, j, np.sqrt(std_x**2 + std_y**2))


if __name__ == '__main__':
    map_data = pgm.read_pgm(map_file)
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
    heatmap = heatmap_gen.compute_heatmap(landmarks, metric='nlls')

    # print(heatmap)

    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max: {}".format(np.max(heatmap)))
    if np.max(heatmap) > 0.0:
        print("heatmap average: {}".format(np.average(heatmap, weights=(heatmap > 0))))

    # Create a mask to dislpay the heatmap on top of the map and transpose for displaying
    heatmap_masked = np.ma.masked_where(map_data == 0, heatmap).transpose()
    plt.imshow(heatmap_masked, 'viridis', interpolation='none', alpha=1.0, origin='lower')
    plt.colorbar()

    # plt.figure()
    # plt.hist(heatmap_masked)
    # plt.figure()
    # plt.imshow(heatmap, cmap='viridis', origin='lower')

    plt.show()
