#!/usr/bin/env python3

""" TODO: docstring """

from time import time
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines

import pgm
import landmark_detection
from heatmap import Heatmap

# Set numpy random seed
np.random.seed(42)

LOG_FILE = "lpo_accuracy.txt"
LANDMARKS_FILE = "landmarks.npy"

TARGET_ACCURACY = 0.1
MAX_INNER_ITER = 500
MAX_LANDMARKS = 15


def plot_configuration(map_data, map_resolution, landmarks, heatmap=None, coverage=None, coverage_score=None, no_show=False):
    """ Display multiple figures for the given landmark configuration """
    map_display = map_data.transpose()
    landmarks_display = landmarks / map_resolution
    plot_something = False
    if heatmap is not None:
        heatmap_masked = np.ma.masked_where(map_data == 0, heatmap).transpose()
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
        coverage_masked = np.ma.masked_where(map_data == 0, coverage).transpose()
        plt.figure()
        plt.imshow(map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m', s=70.0)
        plt.imshow(coverage_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower',
                   vmin=0.0, vmax=landmarks.shape[0])
        plt.tick_params(axis='both', which='major', labelsize=16)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylabel('Coverage (# of landmarks)', rotation=270, fontsize=22, labelpad=25.0)
        triangle = mlines.Line2D([], [], color='m', marker='^', linestyle='None', markersize=10, label='Landmarks')
        legend_handles = [triangle]
        plt.legend(handles=legend_handles, fontsize=20)
        plot_something = True
    if coverage_score is not None:
        score_map_masked = np.ma.masked_where(map_data == 0, coverage_score).transpose()
        plt.figure()
        plt.imshow(map_display, cmap='gray', origin='lower')
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
        plt.imshow(score_map_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower',
                   vmin=0.0, vmax=5.0)
        plt.colorbar()
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
        self.free_cells = np.argwhere(self.free_mask)
        self.land_cells = np.argwhere(~self.free_mask)
        # Precompute which free cells are in range for each land cell
        print("Precomputing cell range...")
        t0 = time()
        self.precompute_range()
        t1 = time()
        print(F"Precompute range time: {t1 - t0}")
        # Initialize heatmap builder
        self.heatmap_builder = Heatmap(map_data, map_resolution)

    def precompute_range(self):
        self.range_coverage_map = np.empty(self.map_data.shape, dtype=object)
        self.cells_in_range = np.zeros((self.map_data.shape[0],
                                        self.map_data.shape[1],
                                        self.map_data.shape[0],
                                        self.map_data.shape[1]),
                                       dtype=bool)
        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            self.range_coverage_map[land_cell_idx] = set()
            land_cell_m = land_cell * self.map_resolution
            for free_cell in self.free_cells:
                free_cell_idx = tuple(free_cell)
                free_cell_m = free_cell * self.map_resolution
                cell_range = np.sqrt((land_cell_m[0] - free_cell_m[0])**2 + (land_cell_m[1] - free_cell_m[1])**2)
                if cell_range < landmark_detection.MAX_RANGE:
                    self.range_coverage_map[land_cell_idx].add(free_cell_idx)
                    self.cells_in_range[land_cell_idx + free_cell_idx] = True
                    self.cells_in_range[free_cell_idx + land_cell_idx] = True

    def get_coverage_map(self, landmarks):
        """ Compute the coverage map, indicating how many landmarks are in range of each cell """
        coverage_map = np.zeros(self.map_data.shape)
        landmark_coords = np.floor_divide(landmarks[:, :2], self.map_resolution).astype('int')
        for (l, landmark_cell) in enumerate(landmark_coords):
            # Check all cells in range of this landmark and increment the coverage on each one
            try:
                for (i, j) in self.range_coverage_map[tuple(landmark_cell[:2])]:
                    coverage_map[i, j] += 1
            except Exception as e:
                print(e)
                print(F"landmark_coords:\n{landmark_coords}")
                print(F"landmark_cell: {landmark_cell}")
                print(F"range_coverage_map: {self.range_coverage_map[tuple(landmark_cell[:2])]}")
        return coverage_map

    # def get_coverage_map(self, landmarks):
    #     """ Compute the coverage map, indicating how many landmarks are in range of each cell """
    #     coverage_map = np.zeros(self.map_data.shape)
    #     for (i, j) in self.free_cells:
    #         cell_t = np.array([i * self.map_resolution, j * self.map_resolution, 0.0])
    #         cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
    #         for l, landmark in enumerate(landmarks):
    #             # Check if landmark is in range
    #             if landmark_detection.distance(cell_pose, landmark) < landmark_detection.MAX_RANGE:
    #                 coverage_map[i, j] += 1
    #     return coverage_map

    def check_coverage(self, landmarks):
        """ Get the coverage map and check if all cells have at least 3 landmarks in range """
        coverage_map = self.get_coverage_map(landmarks)
        # for (i, j) in self.free_cells:
        #     if coverage_map[i, j] < 3:
        #         return False
        # return True
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
            # Check if landmark is in free area
            if self.map_data[landmark_cell_idx] == 255:
                print("Invalid solution: Landmark cell {} is free space!".format(landmark_cell))
                return False
            # Check if landmark is in a cell that is already occupied by another landmark
            if occupied_cells[landmark_cell_idx]:
                print("Invalid solution: Landmark cell {} is already occupied!".format(landmark_cell))
                return False
            occupied_cells[landmark_cell_idx] = True
        # Get the coverage map and check if all cells have at least 3 landmarks in range
        if not self.check_coverage(landmarks):
            # print("Invalid solution: Not all cells are covered by 3 landmarks!")
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

    def find_cell_max_coverage(self, landmarks, randomness=0.0):
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

        for land_cell in self.land_cells:
            land_cell_idx = tuple(land_cell)
            # Do not process land cells where we already have a landmark
            if land_cell_idx in landmark_coords:
                # print("land_cell in landmarks!!!")
                continue
            # Check cells in range of land_cell
            for (i, j) in self.range_coverage_map[land_cell_idx]:
                # Add coverage score to land cell
                if coverage_map[i, j] < 3:
                    land_score[land_cell_idx] += 3 - coverage_map[i, j]

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
        plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
        land_score_masked = np.ma.masked_where(self.map_data == 255, land_score).transpose()
        plt.imshow(land_score_masked, 'plasma', interpolation='none', alpha=1.0, origin='lower')
        plt.colorbar()
        plt.show()

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
        land_score_masked = np.ma.masked_where(self.map_data == 255, land_score).transpose()
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
        with open(LOG_FILE, 'a') as f:
            f.write(F"{n_landmarks},{inner_iter},"
                    F"{best_heatmap_idx},{best_heatmap},"
                    F"{best_coverage_idx},{best_coverage}\n")

    def check_accuracy(self):
        for n in range(self.population_size):
            self.heatmaps[n] = self.heatmap_builder.compute_heatmap(self.population[n], 'nlls')
            self.heatmap_accuracy[n, 0] = self.max_heatmap_accuracy(self.heatmaps[n])
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
        return best_heatmap_accuracy <= TARGET_ACCURACY and self.valid_configuration(self.population[best_heatmap_idx])

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
                   not self.free_mask[new_cell] and \
                   not selected_map[new_cell]:
                    legal_neighbours.append(new_cell)
        return legal_neighbours

    def back_to_school(self):
        """ Allow population to study and exercise, so they can improve and become a better population.
            Local optimization: Landmarks are moved 1 cell to the direction that maximizes coverage.
        """
        for n in range(self.population_size):
            landmark_disorder = np.array(range(self.population.shape[1]))
            np.random.shuffle(landmark_disorder)
            landmark_coords = (self.population[n, :, :2] / self.map_resolution).astype(int)
            for l in landmark_disorder:
                coverage_map = self.get_coverage_map(self.population[n])
                best_fitness = self.fair_coverage_fitness(coverage_map)
                best_landmark = self.population[n, l].copy()
                # Find the legal movements for this landmark
                neighbours = self.find_legal_neighbours(landmark_coords[l], self.population[n])
                for (i, j) in neighbours:
                    neighbour_landmark = np.array([i * self.map_resolution, j * self.map_resolution, 0.0])
                    # Update landmark with neighbour
                    self.population[n, l] = neighbour_landmark
                    # Check neighbour fitness
                    coverage_map = self.get_coverage_map(self.population[n])
                    neighbour_fitness = self.fair_coverage_fitness(coverage_map)
                    # Update landmark position if neighbour has better coverage
                    if neighbour_fitness > best_fitness:
                        best_fitness = neighbour_fitness
                        best_landmark = neighbour_landmark.copy()
                self.population[n, l] = best_landmark
            plt.show()

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
                    new_landmark = self.find_cell_max_coverage(offspring[n, :l, :], randomness=0.3)
                    new_landmark_idx = tuple((new_landmark[:2] / self.map_resolution).astype(int))
                    # Repeat the selection process until we get a landmark position that was not already in the pool
                    while selected_map[new_landmark_idx]:
                        new_landmark = self.find_cell_max_coverage(offspring[n, :l, :], randomness=0.3)
                        new_landmark_idx = tuple((new_landmark[:2] / self.map_resolution).astype(int))
                    offspring[n, l] = new_landmark
                else:
                    # Take it from the list selected from the pool
                    offspring[n, l] = landmark_pool[selected_landmarks[l]]
        return offspring

    def genetic(self):
        """ TODO: docstring """
        self.population_size = 10

        # Find the (theorical) minimum of required landmarks to cover the whole map area
        # Then, add 3 more (because we want to be happy)
        map_area = self.map_data.shape[0] * self.map_data.shape[1] * self.map_resolution**2
        landmark_coverage_area = np.pi * landmark_detection.MAX_RANGE**2
        n_landmarks = int(np.ceil(3 * (map_area / landmark_coverage_area) + 3))
        # n_landmarks = 6

        # Initialize population and fitness arrays
        self.init_population(n_landmarks)
        print(F"Initial population:\n{self.population}")

        # Check accuracy of initial population (maybe we are lucky?)
        if self.check_accuracy():
            best_heatmap_idx, best_heatmap_accuracy = self.get_best_heatmap()
            print("Valid solution found!!")
            print(F"Valid landmarks:\n{self.population[best_heatmap_idx]}")
            return self.population[best_heatmap_idx]

        while True:
            # Outer loop:
            #   Increment number of landmarks until accuracy condition is met
            #   or until all cells have a really high coverage (like 15 landmarks / cell)
            print("#################################################################")
            print(F"[Outer iter] Number of landmarks: {n_landmarks}")
            inner_iter = 0
            while inner_iter < MAX_INNER_ITER:
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
            if n_landmarks > MAX_LANDMARKS:
                raise RuntimeError(F"Could not find a solution after {MAX_LANDMARKS} landmarks")

            print(F"Population after {n_landmarks} landmarks:\n{self.population}")

            # If we still cannot achieve the target accuracy, add another landmark based on greedy algorithm
            # TODO: Greedy algorithm will not work once all cells have ore than 3 landmarks in range.
            #       At that point the chosen cell will be random.
            n_landmarks += 1
            new_landmarks = np.empty((self.population_size, 1, 3))
            for n in range(self.population_size):
                # Extend arrays to make space for an extra landmark
                new_landmarks[n, 0] = self.find_cell_max_coverage(self.population[n], randomness=0.2)
            self.population = np.append(self.population, new_landmarks, axis=1)

    def find_landmarks(self):
        """ Magic """

        # landmarks = self.greedy_fill_coverage()
        landmarks = self.genetic()

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
        return landmarks


def main(args):
    print(F"Log file: {LOG_FILE}")
    print(F"Landmarks file: {LANDMARKS_FILE}")
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

    landmarks = lpo.find_landmarks()
    print("landmarks:\n{}".format(landmarks))

    valid = lpo.valid_configuration(landmarks)
    print("Valid configuration: {}".format(valid))

    # Save the landmarks to a file
    np.save(LANDMARKS_FILE, landmarks)

    heatmap = lpo.heatmap_builder.compute_heatmap(landmarks, 'nlls')
    print("heatmap mean: {}".format(np.mean(heatmap)))
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
    plot_configuration(map_data, RESOLUTION, landmarks, heatmap=heatmap, coverage=coverage)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PGM module test')
    parser.add_argument('map_file', metavar='map_file', type=str,
                        help='Map pgm file')
    parser.add_argument('map_resolution', metavar='map_resolution', type=float,
                        help='Map resolution (m/cell)')
    parser.add_argument('-l', '--landmarks', metavar='landmarks_file', type=str,
                        help='Path to file to save best landmarks (.npy)')
    parser.add_argument('--log', metavar='log_file', type=str,
                        help='Path to log file')
    args = parser.parse_args()

    # Change log file if needed and make path absolute
    if args.log and not os.path.isdir(args.log):
        LOG_FILE = args.log
    LOG_FILE = os.path.abspath(LOG_FILE)

    # Change landmarks file if needed and make path absolute
    if args.landmarks and not os.path.isdir(args.landmarks):
        LANDMARKS_FILE = args.landmarks
    LANDMARKS_FILE = os.path.abspath(LANDMARKS_FILE)

    main(args)
