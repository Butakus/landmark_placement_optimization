#!/usr/bin/env python3

""" TODO: docstring """

from time import time
import ctypes
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray

import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm

import pgm
import metrics
import nlls
import lie_algebra as lie
from landmark_detection import landmark_detection

landmarks = np.array([
    [40.0, 50.0, 0.0],
    [55.0, 25.0, 0.0],
    [65.0, 55.0, 0.0],
    [85.0, 60.0, 0.0],
    [100.0, 30.0, 0.0],
    [120.0, 60.0, 0.0],
    [125.0, 23.0, 0.0],
])

map_file = "/home/butakus/localization_reference/gazebo/map_0p5.pgm"
resolution = 0.5


def map_gdop(map_data):
    gdop_map = np.zeros(map_data.shape)
    for i in tqdm(range(map_data.shape[0])):
        for j in range(map_data.shape[1]):
            # Only map the area if it is drivable
            if map_data[i, j] == 255:
                cell_pose = lie.se3(t=[i * resolution, j * resolution, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
                measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.001)
                try:
                    gdop_map[i][j] = wgdop.compute_gdop(measurements)
                    if gdop_map[i][j] > 250.0:# or np.isnan(gdop_map[i][j]):
                        # print("WWWW!!!")
                        # print("Cell: {}".format(cell_pose))
                        # print("gdop: {}".format(gdop_map[i][j]))
                        gdop_map[i][j] = 250.0
                except np.linalg.LinAlgError:
                    gdop_map[i][j] = 250.0
    return gdop_map

def map_mcmc(map_data):
    mcmc_map = np.zeros(map_data.shape)
    for i in tqdm(range(map_data.shape[0])):
        for j in range(map_data.shape[1]):
            # Only map the area if it is drivable
            if map_data[i, j] == 255:
                cell_t = [i * resolution, j * resolution, 0.0]
                cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
                measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.01)
                nlls_result = nlls.nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                                   initial_guess=None, output=False)
                emcee_result = nlls.mcmc_posterior_estimation(params=nlls_result.params,
                                     args=(landmarks, measurements, measurement_covs),
                                     output=False)
                mcmc_map[i][j] = emcee_result.params['x'].stderr + emcee_result.params['y'].stderr
    return mcmc_map


def map_nlls(map_data):
    nlls_map = np.zeros(map_data.shape)
    for i in tqdm(range(map_data.shape[0])):
        for j in range(map_data.shape[1]):
            # Only map the area if it is drivable
            if map_data[i, j] == 255:
                cell_t = [i * resolution, j * resolution, 0.0]
                cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
                measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.01)
                nlls_result = nlls.nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                                   initial_guess=cell_t, output=False)
                nlls_map[i][j] = nlls_result.params['x'].stderr + nlls_result.params['y'].stderr
    return nlls_map

# def nlls_metric(i, j):
#     cell_t = [i * resolution, j * resolution, 0.0]
#     cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
#     measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.01)
#     nlls_result = nlls.nlls_estimation(args=(landmarks, measurements, measurement_covs),
#                                        initial_guess=cell_t, output=False)
#     std_x = nlls_result.params['x'].stderr
#     std_y = nlls_result.params['y'].stderr
#     # return std_x + std_y
#     return np.sqrt(std_x**2 + std_y**2)

# def mcmc_metric(i, j):
#     cell_t = [i * resolution, j * resolution, 0.0]
#     cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
#     measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.01)
#     nlls_result = nlls.nlls_estimation(args=(landmarks, measurements, measurement_covs),
#                                        initial_guess=None, output=False)
#     emcee_result = nlls.mcmc_posterior_estimation(params=nlls_result.params,
#                                              args=(landmarks, measurements, measurement_covs),
#                                              output=False)
#     std_x = emcee_result.params['x'].stderr
#     std_y = emcee_result.params['y'].stderr
#     # return std_x + std_y
#     return np.sqrt(std_x**2 + std_y**2)

def build_heatmap(map_data, metric):
    """ TODO: docstring """
    heatmap = np.zeros(map_data.shape)
    for i in tqdm(range(map_data.shape[0])):
        for j in range(map_data.shape[1]):
            # Only map the area if it is drivable
            if map_data[i, j] == 255:
                heatmap[i][j] = metric(i, j)
    return heatmap

def test_gdop(x, y):
    X = np.array([x, y, 0.0])
    print("X: {}".format(X))

    # Get the landmark measurements
    X_se3 = lie.se3(t=[X[0], X[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, X[2]]))
    measurements, measurement_covs = landmark_detection(X_se3, landmarks, std=0.0)
    cell_gdop = wgdop.compute_gdop(measurements)
    return cell_gdop


class Heatmap(object):
    """ TODO: docstring for Heatmap """
    def __init__(self, map_data):
        super(Heatmap, self).__init__()
        self.map_data = map_data
        self.heatmap = np.zeros(self.map_data.shape)
        # self.shared_heatmap = RawArray(ctypes.c_double, self.map_data.shape[0] * self.map_data.shape[1])
        self.metrics = {
            'nlls': self.nlls_metric,
            'mcmc': self.mcmc_metric,
        }
        self.results = []

    def add_async_result(self, result):
        # self.results.append(result)
        i, j, val = result
        # shared_heatmap_np = np.frombuffer(self.shared_heatmap,dtype=np.float64).reshape(self.heatmap.shape)
        # shared_heatmap_np[i, j] = val
        self.heatmap[i, j] = val

    def compute_heatmap(self, metric):
        metric_f = self.metrics[metric]
        # pool = mp.Pool(1)
        pool = mp.Pool(mp.cpu_count())
        # heatmap = np.zeros(self.map_data.shape)
        t0 = time()
        for i in range(self.map_data.shape[0]):
            for j in range(self.map_data.shape[1]):
                # Only map the area if it is drivable
                if self.map_data[i, j] == 255:
                    pool.apply_async(metric_f, args=(i, j), callback=self.add_async_result)
                    # async_result = pool.apply_async(metric_f, args=(i, j))
                    # heatmap[i, j] = async_result.get()[2]

        print("Computing heatmap...")
        pool.close()
        pool.join()

        t1 = time()
        print("Heatmap build time: {}".format(t1 - t0))
        # self.heatmap = np.frombuffer(self.shared_heatmap,dtype=np.float64).reshape(self.heatmap.shape)

        # for (i, j, val) in self.results:
        #     heatmap[i, j] = val

        return self.heatmap


    def nlls_metric(self, i, j):
        cell_t = [i * resolution, j * resolution, 0.0]
        cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
        measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.01)
        nlls_result = nlls.nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                           initial_guess=cell_t, output=False)
        std_x = nlls_result.params['x'].stderr
        std_y = nlls_result.params['y'].stderr
        # return std_x + std_y
        return (i, j, np.sqrt(std_x**2 + std_y**2))

    def mcmc_metric(self, i, j):
        cell_t = [i * resolution, j * resolution, 0.0]
        cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
        measurements, measurement_covs = landmark_detection(cell_pose, landmarks, std=0.01)
        nlls_result = nlls.nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                           initial_guess=None, output=False)
        emcee_result = nlls.mcmc_posterior_estimation(params=nlls_result.params,
                                                 args=(landmarks, measurements, measurement_covs),
                                                 output=False)
        std_x = emcee_result.params['x'].stderr
        std_y = emcee_result.params['y'].stderr
        # return std_x + std_y
        return (i, j, np.sqrt(std_x**2 + std_y**2))


        


if __name__ == '__main__':
    map_data = pgm.read_pgm(map_file)
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(resolution))
    print("width: {}".format(width))
    print("height: {}".format(height))

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

    # Compute NLLS posterior:
    # heatmap = build_heatmap(map_data, metric=nlls_metric)
    heatmap_gen = Heatmap(map_data)
    heatmap = heatmap_gen.compute_heatmap(metric='mcmc')

    print(heatmap)
    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max: {}".format(np.max(heatmap)))

    # Create a mask to dislpay the DGOP on top of the map and transpose for displaying
    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap).transpose()
    plt.imshow(heatmap_masked, 'viridis', interpolation='none', alpha=1.0)
    plt.colorbar()

    # plt.figure()
    # plt.hist(heatmap)
    # plt.figure()
    # plt.imshow(heatmap, cmap='viridis', origin='lower')

    plt.show()
