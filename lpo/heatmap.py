#!/usr/bin/env python3

""" TODO: docstring """

from time import time
import multiprocessing as mp
from tqdm import tqdm

import numpy as np
# Set numpy random seed
# np.random.seed(42)

import nlls
import lie_algebra as lie
from landmark_detection import landmark_detection, filter_landmarks, filter_landmarks_occlusions

DEFAULT_NLLS_SAMPLES = 100
N_THREADS = mp.cpu_count()
# N_THREADS = 1

pbar = None

class Heatmap(object):
    """ TODO: docstring for Heatmap """

    def __init__(self, map_data, resolution, nlls_samples=DEFAULT_NLLS_SAMPLES, progress=False):
        super(Heatmap, self).__init__()
        self.map_data = map_data
        self.resolution = resolution
        self.heatmap = np.zeros(self.map_data.shape)
        self.metrics = {
            'nlls': self.nlls_metric,
            'mcmc': self.mcmc_metric,
        }
        self.results = []
        self.landmarks = None
        self.nlls_samples = nlls_samples
        self.progress = progress

    def add_async_result(self, result):
        i, j, val = result
        self.heatmap[i, j] = val
        if self.progress:
            pbar.update()


    def compute_heatmap(self, landmarks, metric):
        global pbar
        self.landmarks = landmarks
        metric_f = self.metrics[metric]
        print("Computing heatmap...")
        if self.progress:
            pbar = tqdm(total=np.count_nonzero(self.map_data == 255))
        t0 = time()
        with mp.get_context("spawn").Pool(N_THREADS) as pool:
            for i in range(self.map_data.shape[0]):
                for j in range(self.map_data.shape[1]):
                    # Only map the area if it is drivable
                    if self.map_data[i, j] == 255:
                        pool.apply_async(metric_f, args=(i, j), callback=self.add_async_result)

            pool.close()
            pool.join()

        t1 = time()
        if self.progress:
            pbar.close()
        print("Heatmap build time: {}".format(t1 - t0))

        return self.heatmap

    def nlls_metric(self, i, j):
        """ TODO: docstring """
        cell_t = np.array([i * self.resolution, j * self.resolution, np.pi/2])
        cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, cell_t[2]]))
        # Get the subset of landmarks that are in range from the current cell
        # filtered_landmarks = filter_landmarks(self.landmarks, cell_pose)
        # filtered_landmarks = filter_landmarks_occlusions(self.landmarks, cell_pose)
        filtered_landmarks = filter_landmarks_occlusions(self.landmarks, cell_pose, self.map_data, self.resolution)
        if filtered_landmarks.shape[0] < 3:
            # Insufficient number of landmarks in range to solve the NLLS problem
            print("WARNING: Not enough landmarks in range!! Cell: {}".format(cell_t))
            return (i, j, np.inf)

        # Take multiple samples to make it more robust
        x = np.zeros(self.nlls_samples)
        y = np.zeros(self.nlls_samples)

        for n in range(self.nlls_samples):
            measurements, measurement_covs = landmark_detection(cell_pose, filtered_landmarks)
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
        for n in range(self.nlls_samples):
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

        measurements, measurement_covs = landmark_detection(cell_pose, filtered_landmarks)

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

