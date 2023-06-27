#!/usr/bin/env python3

""" TODO: docstring """

from time import time
import os
import multiprocessing as mp
import yaml
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

import pgm
import landmark_detection
from heatmap import Heatmap

# Default config parameters
config_params = {
    "map_file": None,
    "map_resolution": None,
    "random_seed": 42,  # Seed for RNG
    "n_threads": mp.cpu_count(),    # Number of threads used

    "log_file": "lpo_accuracy.txt",     # Path to log file
    "landmarks_file": "landmarks.npy",  # Path to output file

    "target_accuracy": 0.015,   # Target std accuracy

    # GA params
    "max_inner_iter": 200,    # Number of max inner iterations
    "max_landmarks": 30,      # Max number of landmarks that can be placed
    "init_num_landmarks": 8,      # Initial number of landmarks
    "population_size": 30,    # Number of particles
    "nlls_samples": 500,      # Number of samples to estimate the localization accuracy
    "heatmap_progress": True, # Show a progressbar for heatmap generation
}

# Set numpy random seed
np.random.seed(config_params["random_seed"])

def read_config(config_file):
    global config_params
    config_params = yaml.safe_load(Path(config_file).read_text())
    # Check params. map_file and map_resolution must exist.
    for t in ['map_file', 'map_resolution']:
        if not t in config_params:
            raise ValueError(F"Configuration file must include the {t} parameter!")

def update_args_params(args):
    global config_params

    config_params['map_file'] = args.map_file
    config_params['map_resolution'] = args.map_resolution
    if args.landmarks and not os.path.isdir(args.landmarks):
        config_params['landmarks_file'] = args.landmarks
    if args.log and not os.path.isdir(args.log):
        config_params['log_file'] = args.log
    if args.particles:
        config_params['population_size'] = args.particles
    if args.samples_nlls:
        config_params['nlls_samples'] = args.samples_nlls
    if args.target_accuracy:
        config_params['target_accuracy'] = args.target_accuracy
    if args.threads:
        config_params['n_threads'] = args.threads
    if args.random_seed:
        config_params['random_seed'] = args.random_seed
    if args.max_inner_iter:
        config_params['max_inner_iter'] = args.max_inner_iter
    if args.max_landmarks:
        config_params['max_landmarks'] = args.max_landmarks
    if args.init_num_landmarks:
        config_params['init_num_landmarks'] = args.init_num_landmarks
    if args.heatmap_progress:
        config_params['heatmap_progress'] = args.heatmap_progress


def plot_configuration(map_data, map_resolution, landmarks, heatmap=None, coverage=None, coverage_score=None, no_show=False):
    """ Display multiple figures for the given landmark configuration """
    map_display = map_data.transpose()
    landmarks_display = landmarks / map_resolution
    plot_something = False
    if heatmap is not None:
        heatmap_masked = np.ma.masked_where(np.logical_or(map_data == 0, map_data == 100), heatmap).transpose()
        plt.figure()
        plt.imshow(map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m', s=70.0)
        plt.imshow(heatmap_masked, 'viridis', interpolation='none', alpha=1.0, origin='lower')
        plt.tick_params(axis='both', which='major', labelsize=16)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Error std (m)', rotation=270, fontsize=22, labelpad=25.0)
        triangle = mlines.Line2D([], [], color='m', marker='^', linestyle='None', markersize=10, label='Landmarks')
        legend_handles = [triangle]
        plt.legend(handles=legend_handles, fontsize=20)
        plot_something = True
    if coverage is not None:
        coverage_masked = np.ma.masked_where(np.logical_or(map_data == 0, map_data == 100), coverage).transpose()
        plt.figure()
        plt.imshow(map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='g', s=70.0)
        # plt.imshow(coverage_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower',
        #            vmin=0.0, vmax=landmarks.shape[0])
        plt.imshow(coverage_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower',
                   vmin=0.0)
        plt.tick_params(axis='both', which='major', labelsize=16)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Coverage (# of landmarks)', rotation=270, fontsize=22, labelpad=25.0)
        triangle = mlines.Line2D([], [], color='g', marker='^', linestyle='None', markersize=10, label='Landmarks')
        legend_handles = [triangle]
        plt.legend(handles=legend_handles, fontsize=20)
        plot_something = True
    if coverage_score is not None:
        score_map_masked = np.ma.masked_where(np.logical_or(map_data == 0, map_data == 100), coverage_score).transpose()
        plt.figure()
        plt.imshow(map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='g')
        plt.imshow(score_map_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower',
                   vmin=0.0, vmax=5.0)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Coverage fitness score', rotation=270, fontsize=22, labelpad=25.0)
        triangle = mlines.Line2D([], [], color='g', marker='^', linestyle='None', markersize=10, label='Landmarks')
        legend_handles = [triangle]
        plt.legend(handles=legend_handles, fontsize=20)
        plot_something = True
    if not plot_something:
        plt.figure()
        plt.imshow(map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
    if not no_show:
        plt.show()


class LPO(object):
    """ TODO: docstring for LPO """

    def __init__(self, map_data, map_resolution):
        super(LPO, self).__init__()
        self.map_data = map_data
        self.map_resolution = map_resolution
        self.map_display = self.map_data.transpose()
        # Precompute free cells and landmark cells
        self.free_mask = self.map_data == 255
        self.land_mask = self.map_data == 100
        self.obstacle_mask = self.map_data == 0
        self.free_cells = np.argwhere(self.free_mask)
        self.land_cells = np.argwhere(self.land_mask)
        self.obstacle_cells = np.argwhere(self.obstacle_mask)
        # Precompute which land cells are visible from each free cell
        print("Precomputing cell visibility...")
        t0 = time()
        self.precompute_visibility()
        t1 = time()
        print(F"Precompute visibility time: {t1 - t0}")
        # Initialize heatmap builder
        self.heatmap_builder = Heatmap(map_data, map_resolution,
                                       nlls_samples=config_params['nlls_samples'],
                                       n_threads=config_params['n_threads'],
                                       progress=config_params['heatmap_progress'])

    def precompute_range(self):
        self.land_visibility_coverage_map = np.empty(self.map_data.shape, dtype=object)
        self.cells_in_range = np.zeros((self.map_data.shape[0],
                                        self.map_data.shape[1],
                                        self.map_data.shape[0],
                                        self.map_data.shape[1]),
                                       dtype=bool)
        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            self.land_visibility_coverage_map[land_cell_idx] = set()
            land_cell_m = land_cell * self.map_resolution
            for free_cell in self.free_cells:
                free_cell_idx = tuple(free_cell)
                free_cell_m = free_cell * self.map_resolution
                cell_range = np.sqrt((land_cell_m[0] - free_cell_m[0])**2 + (land_cell_m[1] - free_cell_m[1])**2)
                if cell_range < landmark_detection.MAX_RANGE:
                    self.land_visibility_coverage_map[land_cell_idx].add(free_cell_idx)
                    self.cells_in_range[land_cell_idx + free_cell_idx] = True
                    self.cells_in_range[free_cell_idx + land_cell_idx] = True

    def precompute_visibility(self):
        # Compute visibility from each free cell
        with mp.get_context("spawn").Pool(config_params['n_threads']) as pool:
            land_visibility_coverage_maps = pool.starmap(self.precompute_cell_visibility, self.free_cells)
            pool.close()
            pool.join()

        # build visibility map
        self.land_visibility_coverage_map = np.empty(self.map_data.shape, dtype=object)
        for i, free_cell in enumerate(self.free_cells):
            free_cell_idx = tuple(free_cell)
            self.land_visibility_coverage_map[free_cell_idx] = land_visibility_coverage_maps[i]

        # Build the inverted visibility map (from land cell to free cells)
        self.land_visibility_coverage_map_inv = np.empty(self.map_data.shape, dtype=object)
        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            self.land_visibility_coverage_map_inv[land_cell_idx] = []
        for free_cell in self.free_cells:
            free_cell_idx = tuple(free_cell)
            for land_cell in self.land_visibility_coverage_map[free_cell_idx]:
                self.land_visibility_coverage_map_inv[land_cell].append(free_cell_idx)


    def precompute_cell_visibility(self, i, j):
        land_visibility_coverage_map = []
        free_cell = np.array([i, j])
        free_cell_idx = (i, j)
        free_cell_m = free_cell * self.map_resolution
        # print("---------------------------------------------------------")
        # print(F"free_cell: {free_cell} / {free_cell_m}")
        # Get all land and obstacle cells in range, and sort them by distance
        occlusion_cells = []
        # First add land cells filtered by distance
        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            land_cell_m = land_cell * self.map_resolution
            cell_distance = landmark_detection.distance_2d(free_cell_m, land_cell_m)
            if cell_distance < landmark_detection.MAX_RANGE:
                # Store distance (for sorting), cell object and boolean to identify land cells
                occlusion_cells.append((cell_distance, land_cell_m, land_cell_idx, True))
        # Then add map obstacles filtered by distance
        for obstacle_cell in self.obstacle_cells:
            obstacle_cell_idx = tuple(obstacle_cell)
            obstacle_cell_m = obstacle_cell * self.map_resolution
            cell_distance = landmark_detection.distance_2d(free_cell_m, obstacle_cell_m)
            if cell_distance < landmark_detection.MAX_RANGE:
                # Store distance (for sorting), obstacle cell object and boolean to identify land cells
                occlusion_cells.append((cell_distance, obstacle_cell_m, obstacle_cell_idx, False))
        # Sort all occlusion cells by distance
        occlusion_cells.sort(key=lambda a: a[0])

        blocked_angles = []
        for cell_distance, occlusion_cell_m, occlusion_cell_idx, is_land_cell in occlusion_cells:
            # Compute occlusion cell angle
            # print("-----------------------")
            # print(F"is_land_cell: {is_land_cell}")
            # print(F"blocked_angles:\n{blocked_angles}")
            # print(F"occlusion_cell: {occlusion_cell_idx} / {occlusion_cell_m}")
            diff = occlusion_cell_m - free_cell_m
            cell_angle = np.arctan2(diff[1], diff[0]) % (2*np.pi)
            # print(F"cell_angle:\n{cell_angle}")
            blocked_angle = False
            if is_land_cell:
                # Check if land cell is blocked by an obstacle and skip it
                for angle_start, alpha in blocked_angles:
                    if landmark_detection.angle_between(cell_angle, angle_start, alpha):
                        # print("Blocked angle")
                        break
                else:
                    # Add land cell to final set
                    # print("Land cell added")
                    land_visibility_coverage_map.append(occlusion_cell_idx)
            else:
                # Compute FOV of obstacle cell and add it to the block list
                alpha = 2 * np.arctan2(self.map_resolution/2, cell_distance)
                angle_start = (cell_angle - alpha/2) % (2*np.pi)
                blocked_angles.append((angle_start, alpha))
        # print(F"Visibility for cell {free_cell_idx}:\n{land_visibility_coverage_map}")
        return land_visibility_coverage_map


    def get_coverage_map_old(self, landmarks):
        """ Compute the coverage map, indicating how many landmarks are in range of each cell """
        coverage_map = np.zeros(self.map_data.shape)
        landmark_coords = np.floor_divide(landmarks[:, :2], self.map_resolution).astype('int')
        for (l, landmark_cell) in enumerate(landmark_coords):
            # Check all cells in range of this landmark and increment the coverage on each one
            try:
                for (i, j) in self.land_visibility_coverage_map[tuple(landmark_cell[:2])]:
                    coverage_map[i, j] += 1
            except Exception as e:
                print(e)
                print(F"landmark_coords:\n{landmark_coords}")
                print(F"landmark_cell: {landmark_cell}")
                print(F"land_visibility_coverage_map: {self.land_visibility_coverage_map[tuple(landmark_cell[:2])]}")
        return coverage_map

    def get_coverage_map(self, landmarks):
        """ Compute the coverage map, indicating how many landmarks are in range of each cell """
        # print("Get coverage!!!!")
        coverage_map = np.zeros(self.map_data.shape)
        landmark_coords = np.floor_divide(landmarks[:, :2], self.map_resolution).astype('int')
        # For each free cell, count the number of visible landmarks (checking occlusions)
        for free_cell in self.free_cells:
            # print("---------------------------------------------------------")
            free_cell_idx = tuple(free_cell)
            free_cell_m = free_cell * self.map_resolution
            # print(F"free_cell: {free_cell} / {free_cell_m}")
            # Get all visible land cells, and sort them by distance
            landmark_cells = []
            # print(F"Land visibility:\n{self.land_visibility_coverage_map[free_cell_idx]}")
            # print("visible landmarks:")
            for (l, landmark_cell) in enumerate(landmark_coords):
                try:
                    if tuple(landmark_cell[:2]) in self.land_visibility_coverage_map[free_cell_idx]:
                        landmark_cell_m = landmark_cell * self.map_resolution
                        # print(F"{landmark_cell} / {landmark_cell_m}")
                        landmark_distance = landmark_detection.distance_2d(free_cell_m, landmark_cell_m)
                        landmark_cells.append((landmark_distance, landmark_cell_m, landmark_cell))
                except Exception as e:
                    print(e)
                    print(F"landmark_coords:\n{landmark_coords}")
                    print(F"landmark_cell: {landmark_cell}")
                    print(F"land_visibility_coverage_map: {self.land_visibility_coverage_map[free_cell_idx]}")
            # print(F"visible landmarks:\n{landmark_cells}")

            # Sort all occlusion cells by distance
            landmark_cells.sort(key=lambda a: a[0])

            # Check which landmarks are actually visible and compute coverage
            blocked_angles = []
            for landmark_distance, landmark_cell_m, landmark_cell_idx in landmark_cells:
                # Compute landmark_cell angle
                # print("----")
                # print(F"blocked_angles:\n{blocked_angles}")
                # print(F"landmark_cell_m:\n{landmark_cell_m}")
                diff = landmark_cell_m - free_cell_m
                landmark_angle = np.arctan2(diff[1], diff[0]) % (2*np.pi)
                # print(F"landmark_angle:\n{landmark_angle}")
                # Check if landmark is blocked by a closer landmark and skip it
                for angle_start, alpha in blocked_angles:
                    if landmark_detection.angle_between(landmark_angle, angle_start, alpha):
                        # print("BLOCKED")
                        break
                else:
                    # Increment coverage for current free cell
                    coverage_map[free_cell_idx] += 1
                    # Compute FOV of new landmark and add it to the block list
                    alpha = 2 * np.arctan2(landmark_detection.POLE_RADIUS, landmark_distance)
                    angle_start = (landmark_angle - alpha/2) % (2*np.pi)
                    blocked_angles.append((angle_start, alpha))
        return coverage_map


    def check_coverage(self, landmarks):
        """ Get the coverage map and check if all cells have at least 3 landmarks in range """
        coverage_map = self.get_coverage_map(landmarks)
        return not (coverage_map[self.free_mask] < 3).any()

    def valid_configuration(self, landmarks):
        """ Check if the landmark configuration is valid
            - All free cells are covered by at least 3 landmarks.
            - Landmarks are only in landmark-area (non-free).
            - Each non-free cell contains at most one landmark.
        """
        # Get the cell coordinates of the landmarks
        landmark_coords = np.floor_divide(landmarks[:, :2], self.map_resolution).astype('int')

        occupied_cells = np.zeros(self.map_data.shape, dtype=bool)
        for (l, landmark_cell) in enumerate(landmark_coords):
            landmark_cell_idx = tuple(landmark_cell)
            # Check if landmark is outside of map boundaries
            if landmark_cell[0] < 0 or landmark_cell[0] >= self.map_data.shape[0] or\
               landmark_cell[1] < 0 or landmark_cell[1] >= self.map_data.shape[1]:
                print("Invalid solution: Landmark {} is out of map range!".format(landmarks[l]))
                return False
            # Check if landmark is in land area
            if self.map_data[landmark_cell_idx] != 100:
                print("Invalid solution: Landmark cell {} is not in land space!".format(landmark_cell))
                return False
            # Check if landmark is in a cell that is already occupied by another landmark
            if occupied_cells[landmark_cell_idx]:
                print("Invalid solution: Landmark cell {} is already occupied!".format(landmark_cell))
                return False
            occupied_cells[landmark_cell_idx] = True
        # Get the coverage map and check if all cells have at least 3 landmarks in range
        if not self.check_coverage(landmarks):
            print("Invalid solution: Not all cells are covered by 3 landmarks!")
            return False
        return True

    def max_heatmap_accuracy(self, heatmap):
        """ Cost function based on the coverage map
            Returns a fitness score based on the minimum coverage value
        """
        return np.max(heatmap[self.free_mask])

    def min_coverage_fitness(self, coverage_map):
        """ Cost function based on the coverage map
            Returns a fitness score based on the minimum coverage value
        """
        return np.min(coverage_map[self.free_mask])

    def min_count_coverage_fitness(self, coverage_map):
        """ Cost function based on the coverage map
            Returns a fitness score based on how many cells are with minimum coverage
        """
        return np.count_nonzero(coverage_map[self.free_mask] == np.min(coverage_map[self.free_mask]))

    def sum_coverage_fitness(self, coverage_map):
        """ Cost function based on the coverage map
            Returns a fitness score based on the coverage of each cell
        """
        return np.sum(coverage_map[self.free_mask])

    def fair_coverage_fitness(self, coverage_map):
        """ Cost function based on the coverage map
            Returns a fitness score based on the coverage of each cell
        """
        score_map = coverage_map.copy()
        score_map[self.free_mask] = 5 * np.tanh(score_map[self.free_mask] / 3)
        return np.sum(score_map[self.free_mask])

    def squared_coverage_fitness(self, coverage_map):
        """ Cost function based on the coverage map
            Returns a fitness score based on the coverage of each cell
        """
        return np.sqrt(np.sum(coverage_map**2))

    def find_cell_max_coverage(self, landmarks, randomness=0.0, blocked_coords=[]):
        """ Find the free cell that maximizes the coverage after placing a new landmark there
            The score of each cell is determined by the amount of landmarks in range:
                0  in range --> 3 points
                1  in range --> 2 points
                2  in range --> 1 points
                >3 in range --> 0 points
            Then, one of the cells with the highest score is randomly selected to place a new landmark
        """
        if randomness < 0.0 or randomness > 1.0:
            print("WARNING: Randomness argument must be a number in the range [0.0, 1.0]")
            randomness = np.clip(randomness, 0.0, 1.0)

        coverage_map = self.get_coverage_map(landmarks)
        land_score = np.zeros(self.map_data.shape)

        landmark_coords = np.floor_divide(landmarks[:, :2], self.map_resolution).astype('int')
        landmark_coords = [tuple(c) for c in landmark_coords]
        blocked_coords = [tuple(c) for c in blocked_coords]

        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            # Do not process land cells where we already have a landmark
            if land_cell_idx in landmark_coords or land_cell_idx in blocked_coords:
                # print("land_cell in landmarks!!!")
                land_score[land_cell_idx] = -1
                continue
            # Check cells in range of land_cell
            for (i, j) in self.land_visibility_coverage_map_inv[land_cell_idx]:
                # Add coverage score to land cell
                if coverage_map[i, j] < 3:
                    land_score[land_cell_idx] += 3 - coverage_map[i, j]

        # Hide free cells so they cannot be chosen (even if max_score is zero)
        land_score[self.free_mask | self.obstacle_mask] = -1

        # Select randomly one of the land cells with the highest scores
        max_score = np.max(land_score)

        threshold = int((1 - randomness)*max_score)
        best_scores = np.argwhere(land_score >= threshold)
        new_landmark_cell = best_scores[np.random.choice(best_scores.shape[0])]

        new_landmark = np.array([new_landmark_cell[0] * self.map_resolution, new_landmark_cell[1] * self.map_resolution, 0.0])

        # Create a mask to dislpay the score map on top of the map and transpose for displaying
        # landmarks_display = landmarks / self.map_resolution
        # plt.imshow(self.map_display, cmap='gray', origin='lower')
        # plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
        # land_score_masked = np.ma.masked_where(np.logical_or(self.map_data == 0, self.map_data == 255), land_score).transpose()
        # plt.imshow(land_score_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower')
        # plt.colorbar()
        # plt.show()

        return new_landmark

    def find_cell_max_coverage_2(self, landmarks, randomness=0.0):
        """ Find the free cell that maximizes the coverage after placing a new landmark there
            The score of each cell is determined by the coverage score after placing the new landmark there
            Then, one of the cells with the highest score is randomly selected to place a new landmark
        """
        if randomness < 0.0 or randomness > 1.0:
            print("WARNING: Randomness argument must be a number in the range [0.0, 1.0]")
            randomness = np.clip(randomness, 0.0, 1.0)

        coverage_map = self.get_coverage_map(landmarks)
        land_score = np.zeros(self.map_data.shape)

        landmark_coords = np.floor_divide(landmarks[:, :2], self.map_resolution).astype('int')
        landmark_coords = [tuple(c) for c in landmark_coords]
        min_score = np.inf
        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            # Do not process land cells where we already have a landmark
            if land_cell_idx in landmark_coords:
                # print("land_cell in landmarks!!!")
                continue
            # Build a new list of landmarks and check what would be the coverage score with the new addition
            temp_landmark = np.array([land_cell[0] * self.map_resolution, land_cell[1] * self.map_resolution, 0.0])
            temp_landmarks = np.concatenate((landmarks, np.array([temp_landmark])), axis=0)
            coverage_map = self.get_coverage_map(temp_landmarks)
            land_score[land_cell_idx] = self.fair_coverage_fitness(coverage_map)
            min_score = min(min_score, land_score[land_cell_idx])

        # Hide free cells so they cannot be chosen (even if max_score is zero)
        land_score[self.free_mask] = -1

        # Select randomly one of the land cells with the highest scores
        max_score = np.max(land_score)

        best_scores = np.argwhere(land_score >= int((1 - randomness)*max_score))
        new_landmark_cell = best_scores[np.random.choice(best_scores.shape[0])]

        new_landmark = np.array([new_landmark_cell[0] * self.map_resolution, new_landmark_cell[1] * self.map_resolution, 0.0])

        # Create a mask to dislpay the score map on top of the map and transpose for displaying
        landmarks_display = landmarks / self.map_resolution
        plt.imshow(self.map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='tab:green', s=150.0)
        land_score_masked = np.ma.masked_where(np.logical_or(self.map_data == 0, self.map_data == 255), land_score).transpose()
        plt.imshow(land_score_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower',
                   vmin=min_score, vmax=max_score)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        triangle = mlines.Line2D([], [], color='tab:green', marker='^',
                                 linestyle='None', markersize=10, label='Landmarks')
        legend_handles = [triangle]
        plt.legend(handles=legend_handles, fontsize=20)
        plt.show()

        return new_landmark

    def greedy_fill_coverage(self):
        """ Use a greedy algorithm that places landmarks with max coverage until all cells are covered """
        landmarks = np.empty((0, 3))
        # Find the free cell that maximizes the coverage
        while not self.check_coverage(landmarks):
            new_landmark = self.find_cell_max_coverage(landmarks, randomness=0.1)
            landmarks = np.concatenate((landmarks, np.array([new_landmark])), axis=0)
        return landmarks

    def get_best_heatmap(self):
        """ Find the index of the heatmap with the minimum error.
            Return the index and the error value
        """
        best_heatmap_idx = np.argmin(self.heatmap_accuracy[:, 0])
        best_heatmap_accuracy = self.heatmap_accuracy[best_heatmap_idx, 0]
        return (best_heatmap_idx, best_heatmap_accuracy)

    def save_best_accuracy(self, n_landmarks, inner_iter):
        """ Write the best heatmap accuracy (and coverage fitness) into a file in CSV format:
            "Number of landmarks, current inner iteration, heatmap index, heatmap value, coverage index, coverage value"
        """
        best_heatmap_idx, best_heatmap = self.get_best_heatmap()
        best_coverage_idx = np.argmax(self.coverage_fitness[:, 0])
        best_coverage = self.coverage_fitness[best_coverage_idx, 0]
        best_landmarks = self.population[best_heatmap_idx]
        with open(config_params['log_file'], 'a') as f:
            f.write(F"{n_landmarks},{inner_iter},"
                    F"{best_heatmap_idx},{best_heatmap},"
                    F"{best_coverage_idx},{best_coverage}\n")
        # Save temporary set of landmarks
        landmarks_temp_file = config_params['landmarks_file'].rstrip(".npy") + F"_{n_landmarks}_{inner_iter}.npy"
        np.save(landmarks_temp_file, best_landmarks)
        print(F"Temporary landmark set saved to file: {landmarks_temp_file}")

    def check_accuracy(self):
        for n in range(self.population_size):
            if self.valid_configuration(self.population[n]):
                self.heatmaps[n] = self.heatmap_builder.compute_heatmap(self.population[n], 'nlls')
                self.heatmap_accuracy[n, 0] = self.max_heatmap_accuracy(self.heatmaps[n])
            else:
                self.heatmaps[n] = None
                self.heatmap_accuracy[n, 0] = np.inf
            print(F"Heatmap accuracy: {self.heatmap_accuracy[n, 0]}")
            # Display stuff
            if False:
                self.coverage_maps[n] = self.get_coverage_map(self.population[n])
                plot_configuration(self.map_data, self.map_resolution,
                                   self.population[n],
                                   heatmap=self.heatmaps[n],
                                   coverage=self.coverage_maps[n])
        print(F"Heatmap fitness:\n{self.heatmap_accuracy}")
        best_heatmap_idx, best_heatmap_accuracy = self.get_best_heatmap()
        print(F"Best accuracy: {best_heatmap_accuracy}")
        return best_heatmap_accuracy <= config_params['target_accuracy'] and self.valid_configuration(self.population[best_heatmap_idx])

    def init_population(self, n_landmarks):
        """ Initialize population and fitness arrays """
        self.population = np.empty((self.population_size, n_landmarks, 3))
        self.coverage_maps = np.empty((self.population_size, self.map_data.shape[0], self.map_data.shape[1]))
        self.heatmaps = np.empty((self.population_size, self.map_data.shape[0], self.map_data.shape[1]))
        self.heatmap_accuracy = np.empty((self.population_size, 2))
        self.heatmap_accuracy[:, 1] = range(self.population_size)
        self.coverage_fitness = np.empty((self.population_size, 2))
        self.coverage_fitness[:, 1] = range(self.population_size)
        for n in range(self.population_size):
            for l in range(n_landmarks):
                self.population[n, l] = self.find_cell_max_coverage(self.population[n, :l, :], randomness=0.2)
            self.coverage_maps[n] = self.get_coverage_map(self.population[n])
            self.coverage_fitness[n, 0] = self.fair_coverage_fitness(self.coverage_maps[n])
            self.heatmaps[n] = None
            self.heatmap_accuracy[n, 0] = np.inf

            # Display stuff
            if False:
                self.heatmaps[n] = self.heatmap_builder.compute_heatmap(self.population[n], 'nlls')
                self.heatmap_accuracy[n, 0] = self.max_heatmap_accuracy(self.heatmaps[n])
                score_map = self.coverage_maps[n].copy()
                score_map[self.free_mask] = 5 * np.tanh(score_map[self.free_mask] / 3)
                plot_configuration(self.map_data, self.map_resolution,
                                   self.population[n],
                                   heatmap=self.heatmaps[n],
                                   coverage=self.coverage_maps[n],
                                   coverage_score=score_map)

    def find_legal_neighbours(self, landmark_cell, landmarks):
        """ Find the legal neighbours of a landmark.
            A valid neighbour must be land space within the map range.
            Also, a valid neighbour must not contain a landmark.
        """
        # Build a map of the other landmarks to avoid stepping on them
        selected_map = np.zeros(self.map_data.shape, dtype=bool)
        landmark_coords = (landmarks[:, :2] / self.map_resolution).astype(int)
        selected_map[tuple(landmark_coords.T)] = True

        legal_neighbours = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                new_cell = (landmark_cell[0] + dx, landmark_cell[1] + dy)
                if (dx != 0 or dy != 0) and \
                   new_cell[0] > 0 and new_cell[1] > 0 and \
                   new_cell[0] < self.map_data.shape[0] and \
                   new_cell[1] < self.map_data.shape[1] and \
                   self.land_mask[new_cell] and \
                   not selected_map[new_cell]:
                    legal_neighbours.append(new_cell)
        return legal_neighbours

    def back_to_school(self):
        """ Allow population to study and exercise, so they can improve and become a better population.
            Local optimization: Landmarks are moved 1 cell to the direction that maximizes coverage.
        """
        with mp.get_context("spawn").Pool(config_params['n_threads']) as pool:
            better_population_list = pool.map(self.back_to_school_element, list(range(self.population_size)))
            pool.close()
            pool.join()
            self.population = np.array(better_population_list)

    def back_to_school_element(self, n):
        """ Allow population to study and exercise, so they can improve and become a better population.
            Local optimization: Landmarks are moved 1 cell to the direction that maximizes coverage.
        """
        better_population = self.population[n]
        landmark_disorder = np.array(range(self.population.shape[1]))
        np.random.shuffle(landmark_disorder)
        landmark_coords = (self.population[n, :, :2] / self.map_resolution).astype(int)
        for l in landmark_disorder:
            coverage_map = self.get_coverage_map(better_population)
            best_fitness = self.fair_coverage_fitness(coverage_map)
            best_landmark = better_population[l].copy()
            # Find the legal movements for this landmark
            neighbours = self.find_legal_neighbours(landmark_coords[l], better_population)
            for (i, j) in neighbours:
                neighbour_landmark = np.array([i * self.map_resolution, j * self.map_resolution, 0.0])
                # Update landmark with neighbour
                better_population[l] = neighbour_landmark
                # Check neighbour fitness
                coverage_map = self.get_coverage_map(better_population)
                neighbour_fitness = self.fair_coverage_fitness(coverage_map)
                # Update landmark position if neighbour has better coverage
                if neighbour_fitness > best_fitness:
                    best_fitness = neighbour_fitness
                    best_landmark = neighbour_landmark.copy()
            better_population[l] = best_landmark
        return np.array(better_population)

    def tournament(self, winner_indices):
        """ Tournament process. Get a random subset of the population and return the best element """
        # Filter indices which are already selected
        candidates = np.array(range(self.population_size))[~winner_indices]
        # Each round in the tournament will pick up to a 50% of the population
        tournament_size = int(round(0.5 * self.population_size))
        # Adjust the tournament size if we are running out of candidates
        tournament_size = min(tournament_size, candidates.shape[0])
        # Pick the selected elements indices
        selected = np.random.choice(candidates, tournament_size, replace=False)
        # Winner index (in selected array)
        winner_selected = np.argmax(self.coverage_fitness[selected, 0])
        # Actual winner index (in global population array)
        winner_global = self.coverage_fitness[selected[winner_selected], 1]
        return int(winner_global)

    def selection(self):
        """ Return the best elements from the current population (by tournament) """
        # 20% of the new population will be elements from the current population
        number_of_winners = int(round(0.3 * self.population_size))
        winners = np.zeros(number_of_winners, dtype=int)
        winner_indices = np.zeros(self.population_size, dtype=bool)
        # But the best element will automatically pass (VIP access)
        vip_idx = np.argmax(self.coverage_fitness[:, 0])
        winners[0] = vip_idx
        winner_indices[vip_idx] = True
        for i in range(1, number_of_winners):
            winner_idx = self.tournament(winner_indices)
            winners[i] = winner_idx
            winner_indices[winner_idx] = True
        return self.population[winners]

    def offspring_generation(self, winners):
        """ Generate offspring from parents and add mutation to offspring.
            The offspring is created from ALL parents (orgy mode):
              1. Create a pool of good landmark positions from all parents
              2. Select the children landmarks from the pool
              3. If a children landmark is mutated, it will be drawn from greedy alg, with 30% randomness
        """
        mutation_rate = 0.2
        offspring_size = self.population_size - winners.shape[0]
        offspring = np.zeros((offspring_size, winners.shape[1], 3))
        # Collect all landmark positions from all winners
        placement_selected = np.zeros(self.map_data.shape, dtype=bool)
        landmark_pool = []
        landmark_pool_coords = []
        for winner in winners:
            for landmark in winner:
                landmark_idx = tuple((landmark[:2] / self.map_resolution).astype(int))
                if not placement_selected[landmark_idx]:
                    placement_selected[landmark_idx] = True
                    landmark_pool.append(landmark)
                    landmark_pool_coords.append(landmark_idx)
        landmark_pool_coords = np.array(landmark_pool_coords)

        # plot_configuration(self.map_data, self.map_resolution, np.array(landmark_pool))

        for n in range(offspring.shape[0]):
            selected_landmarks = np.random.choice(len(landmark_pool), offspring.shape[1], replace=False)
            selected_map = np.zeros(self.map_data.shape, dtype=bool)
            selected_map[tuple(landmark_pool_coords[selected_landmarks].T)] = True
            for l in range(offspring.shape[1]):
                # Check if the landmark is mutated, or drawn from the parents pool
                if np.random.random() < mutation_rate:
                    # Mutate landmark. Select new landmark position with greedy algorithm
                    new_landmark = self.find_cell_max_coverage(offspring[n, :l, :], randomness=0.3, blocked_coords=landmark_pool_coords[selected_landmarks])
                    offspring[n, l] = new_landmark
                else:
                    # Take it from the list selected from the pool
                    offspring[n, l] = landmark_pool[selected_landmarks[l]]
        return offspring

    def genetic(self):
        """ TODO: docstring """
        self.population_size = config_params['population_size']

        # Initial number of landmarks can be obtained from param or from an area-based estimation by default
        n_landmarks = 1
        if 'init_num_landmarks' in config_params:
            n_landmarks = int(np.ceil(3 * (map_area / landmark_coverage_area) + config_params['init_num_landmarks']))
        else:
            # Find the (theorical) minimum of required landmarks to cover the whole map area
            map_area = self.map_data.shape[0] * self.map_data.shape[1] * self.map_resolution**2
            landmark_coverage_area = np.pi * landmark_detection.MAX_RANGE**2
            n_landmarks = int(np.ceil(3 * map_area / landmark_coverage_area))

        # Initialize population and fitness arrays
        self.init_population(n_landmarks)
        print(F"Initial population:\n{self.population}")

        # Check accuracy of initial population (maybe we are lucky?)
        if self.check_accuracy():
            best_heatmap_idx, best_heatmap_accuracy = self.get_best_heatmap()
            print("Valid solution found!!")
            print(F"Valid landmarks:\n{self.population[best_heatmap_idx]}")
            # Save to log file
            self.save_best_accuracy(n_landmarks, 0)
            return self.population[best_heatmap_idx]

        while True:
            # Outer loop:
            #   Increment number of landmarks until accuracy condition is met
            #   or until all cells have a really high coverage (like 15 landmarks / cell)
            print("#################################################################")
            print(F"[Outer iter] Number of landmarks: {n_landmarks}")
            inner_iter = 0
            while inner_iter < config_params['max_inner_iter']:
                # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print(F"[Inner iter] {inner_iter}")
                # Inner loop.
                #   Genetic iterations until MAX_ITER is reached or until accuracy condition is met

                # Local optimization: Landmarks are moved 1 cell to the direction that maximizes coverage
                self.back_to_school()

                # Update population fitness (only coverage)
                for n in range(self.population_size):
                    self.coverage_maps[n] = self.get_coverage_map(self.population[n])
                    self.coverage_fitness[n, 0] = self.fair_coverage_fitness(self.coverage_maps[n])

                # Select winners (parents) based on tournament selection
                winners = self.selection()

                # Generate offspring from parents and add mutation to offspring
                offspring = self.offspring_generation(winners)
                # Update population with parents + offspring
                self.population[:winners.shape[0]] = winners
                self.population[winners.shape[0]:] = offspring

                inner_iter += 1

                # Check heatmap accuracy every 100 iterations
                if inner_iter % 100 == 0:
                    if self.check_accuracy():
                        best_heatmap_idx, best_heatmap_accuracy = self.get_best_heatmap()
                        print("Valid solution found!!")
                        print(F"Valid landmarks:\n{self.population[best_heatmap_idx]}")
                        return self.population[best_heatmap_idx]
                    # Save to file
                    self.save_best_accuracy(n_landmarks, inner_iter)

            # TODO: Break point. Do this better.
            if n_landmarks > config_params['max_landmarks']:
                raise RuntimeError(F"Could not find a solution after {config_params['max_landmarks']} landmarks")

            print(F"Population after {n_landmarks} landmarks:\n{self.population}")

            # If we still cannot achieve the target accuracy, add another landmark based on greedy algorithm
            # TODO: Greedy algorithm will not work once all cells have more than 3 landmarks in range.
            #       At that point the chosen cell will be random.
            n_landmarks += 1
            new_landmarks = np.empty((self.population_size, 1, 3))
            for n in range(self.population_size):
                # Extend arrays to make space for an extra landmark
                new_landmarks[n, 0] = self.find_cell_max_coverage(self.population[n], randomness=0.2)
            self.population = np.append(self.population, new_landmarks, axis=1)


    def clean_solution(self, landmarks):
        """ Take a final and valid solution and try to simplify it by removing unnecessary landmarks """
        print("Cleaning solution...")
        print(F"Landmarks:\n{landmarks}")

        # Iterate over all landmarks in random order
        landmark_disorder = np.array(range(landmarks.shape[0]))
        np.random.shuffle(landmark_disorder)
        for i in landmark_disorder:
            # Remove landmark i
            print(F"Checking landmark {landmarks[i]}...")
            clean_landmarks = np.delete(landmarks, i, axis=0)
            # Check if solution is valid (coverage might be lost)
            if self.valid_configuration(clean_landmarks):
                print("Solution is still valid")
                # Compute heatmap and check if accuracy is in range
                heatmap = self.heatmap_builder.compute_heatmap(clean_landmarks, 'nlls')
                heatmap_accuracy = self.max_heatmap_accuracy(heatmap)
                print(F"Heatmap accuracy: {heatmap_accuracy}")
                if heatmap_accuracy < config_params['target_accuracy']:
                    print(F"Landmark {landmarks[i]} can be removed from solution")
                    print(F"New heatmap accuracy: {heatmap_accuracy}")
                    return self.clean_solution(clean_landmarks)
                else:
                    print(F"Landmark {landmarks[i]} seems important for accuracy")
            else:
                print(F"Solution not valid after removing landmark {landmarks[i]}")
        print("Current solution cannot be reduced any more")
        return landmarks



    def find_landmarks(self):
        """ Magic """

        landmarks = self.genetic()

        """ Below are "manual" solutions to test other features without
            having to find the best landmarks, which is an expensive task
        """
        # landmarks = self.greedy_fill_coverage()

        # landmarks = np.array([
        #     # [5.0, 40.0, 0.0],
        #     [40.0, 45.0, 0.0],
        #     [55.0, 25.0, 0.0],
        #     [65.0, 55.0, 0.0],
        #     [85.0, 60.0, 0.0],
        #     [100.0, 30.0, 0.0],
        #     [120.0, 60.0, 0.0],
        #     [125.0, 23.0, 0.0],
        # ])

        # R4
        # landmarks = np.array([
        #     [0., 0., 0.],
        #     [0., 32., 0.],
        #     [8., 8., 0.],
        #     [12., 36., 0.],
        #     [16., 0., 0.],
        #     [28., 8., 0.],
        #     [28., 28., 0.],
        #     [32., 36., 0.],
        #     [36., 0., 0.],
        #     [48., 32., 0.],
        #     [52., 32., 0.],
        #     [64., 0., 0.],
        #     [76., 4., 0.],
        #     [76., 16., 0.],
        #     [76., 32., 0.],
        # ])

        # R1 / R2
        # landmarks = np.array([
        #     [0., 0., 0.],
        #     [0., 32., 0.],
        #     [10., 12., 0.],
        #     [12., 40., 0.],
        #     [16., 0., 0.],
        #     [16., 48., 0.],
        #     [28., 12., 0.],
        #     [28., 30., 0.],
        #     [32., 40., 0.],
        #     [36., 0., 0.],
        #     [48., 36., 0.],
        #     [52., 36., 0.],
        #     [64., 0., 0.],
        #     [68., 48., 0.],
        #     [78., 4., 0.],
        #     [78., 16., 0.],
        #     [78., 32., 0.],
        # ])

        # "Valid" solution found for R4 with low NLLS samples
        # landmarks = np.array([
        #     [32., 36.,  0.],
        #     [76., 12.,  0.],
        #     [ 0., 12.,  0.],
        #     [48., 32.,  0.],
        #     [20., 44.,  0.],
        #     [36., 44.,  0.],
        #     [76., 32.,  0.],
        #     [ 8., 12.,  0.],
        #     [20., 36.,  0.],
        #     [ 0., 16.,  0.],
        #     [76., 16.,  0.],
        #     [28.,  0.,  0.],
        #     [56.,  0.,  0.],
        #     [ 8.,  8.,  0.],
        #     [12.,  0.,  0.],
        # ])
        return landmarks


def main():
    print(F"Log file: {config_params['log_file']}")
    print(F"Landmarks file: {config_params['landmarks_file']}")
    map_data = pgm.read_pgm(config_params['map_file'])
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(config_params['map_resolution']))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width*height))
    print("Map free cells: {}".format(np.count_nonzero(map_data == 255)))

    # Find a landmark setup that guarantees the desired accuracy
    lpo = LPO(map_data, config_params['map_resolution'])

    landmarks = lpo.find_landmarks()
    print("landmarks:\n{}".format(landmarks))

    valid = lpo.valid_configuration(landmarks)
    print("Valid configuration: {}".format(valid))

    # Try to clean the solution and remove unnecessary landmarks
    landmarks = lpo.clean_solution(landmarks)

    # Save the landmarks to a file
    np.save(config_params['landmarks_file'], landmarks)

    # heatmap = None
    heatmap = lpo.heatmap_builder.compute_heatmap(landmarks, 'nlls')
    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max: {}".format(np.max(heatmap)))
    if np.max(heatmap) > 0.0:
        print("heatmap average: {}".format(np.average(heatmap, weights=(heatmap > 0))))

    coverage = lpo.get_coverage_map(landmarks)
    print("coverage:\n{}".format(coverage))
    print("coverage mean: {}".format(np.mean(coverage)))
    print("coverage max: {}".format(np.max(coverage)))
    if np.max(coverage) > 0.0:
        print("coverage average: {}".format(np.average(coverage, weights=(coverage > 0))))
    print("coverage fitness: {}".format(lpo.fair_coverage_fitness(coverage)))

    # Display the maps for the obtained landmark configuration
    plot_configuration(map_data, config_params['map_resolution'], landmarks, heatmap=heatmap, coverage=coverage)


if __name__ == '__main__':
    import argparse
    conf_parser = argparse.ArgumentParser(
        description='Landmark Placement Optimization module',
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Turn off help, so we print all options in response to -h
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file",
                             help="Path to configuration yaml file",
                             type=str, metavar="config_file")
    args, remaining_argv = conf_parser.parse_known_args()

    # Extract and check parameters from config file
    if args.conf_file:
        print(F"config_file: {args.conf_file}")
        read_config(args.conf_file)
    else:
        parser = argparse.ArgumentParser(parents=[conf_parser])
        parser.add_argument('map_file', metavar='map_file', type=str,
                            help='Map pgm file')
        parser.add_argument('map_resolution', metavar='map_resolution', type=float,
                            help='Map resolution (m/cell)')
        parser.add_argument('-l', '--landmarks', metavar='landmarks_file', type=str,
                            help='Path to file to save best landmarks (.npy)',
                            default=config_params['landmarks_file'])
        parser.add_argument('--log', metavar='log_file', type=str,
                            help='Path to log file', default=config_params['log_file'])
        parser.add_argument('-p', '--particles', metavar='population_size', type=int,
                            help='GA number of particles', default=config_params['population_size'])
        parser.add_argument('-s', '--samples-nlls', metavar='samples_nlls', type=int,
                            help='Number of samples to estimate accuracy',
                            default=config_params['nlls_samples'])
        parser.add_argument('-a', '--target-accuracy', metavar='target_accuracy', type=float,
                            help='Target std accuracy', default=config_params['target_accuracy'])
        parser.add_argument('-t', '--threads', metavar='n_threads', type=int,
                            help='Number of threads', default=config_params['n_threads'])
        parser.add_argument('--random-seed', metavar='random_seed', type=int,
                            help='RNG seed', default=config_params['random_seed'])
        parser.add_argument('--max-inner-iter', metavar='max_inner_iter', type=int,
                            help='Max number of inner iterations',
                            default=config_params['max_inner_iter'])
        parser.add_argument('--max-landmarks', metavar='max_landmarks', type=int,
                            help='Max number of landmarks to place',
                            default=config_params['max_landmarks'])
        parser.add_argument('--init-num-landmarks', metavar='init_num_landmarks', type=int,
                            help='Initial number of landmarks to try',
                            default=config_params['init_num_landmarks'])
        parser.add_argument('--heatmap-progress', metavar='heatmap_progress', type=bool,
                            help='Show heatmap progress bar',
                            default=config_params['heatmap_progress'])

        remaining_args = parser.parse_args(remaining_argv)
        update_args_params(remaining_args)

    print("Config params:")
    for k,v in config_params.items():
        print(F"\t* {k}: {v}")

    main()
