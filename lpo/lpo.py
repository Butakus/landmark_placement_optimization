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
import lie_algebra as lie
import landmark_detection
import nlls
from heatmap import Heatmap

map_file = "/home/butakus/localization_reference/gazebo/map_5p0.pgm"
resolution = 5.0

target_accuracy = 0.1


class LPO(object):
    """ TODO: docstring for LPO """
    def __init__(self, map_data):
        super(LPO, self).__init__()
        self.map_data = map_data
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
        self.heatmap_builder = Heatmap(map_data, resolution)

    def precompute_range(self):
        self.range_coverage_map = np.empty(self.map_data.shape, dtype=object)
        self.cells_in_range = np.zeros((self.map_data.shape[0],
                                        self.map_data.shape[1],
                                        self.map_data.shape[0],
                                        self.map_data.shape[1]),
                                        dtype=bool)
        for land_cell in self.land_cells:
            # print(F"land_cell: {land_cell}")
            land_cell_idx = tuple(land_cell)
            self.range_coverage_map[land_cell_idx] = set()
            land_cell_m = land_cell * resolution
            for free_cell in self.free_cells:
                free_cell_idx = tuple(free_cell)
                free_cell_m = free_cell * resolution
                cell_range = np.sqrt((land_cell_m[0] - free_cell_m[0])**2 + (land_cell_m[1] - free_cell_m[1])**2)
                if cell_range < landmark_detection.MAX_RANGE:
                    self.range_coverage_map[land_cell_idx].add(free_cell_idx)
                    self.cells_in_range[land_cell_idx + free_cell_idx] = True
                    self.cells_in_range[free_cell_idx + land_cell_idx] = True
            # print(F"range_coverage_map:\n{self.range_coverage_map[land_cell_idx]}")


    def get_coverage_map(self, landmarks):
        """ Compute the coverage map, indicating how many landmarks are in range of each cell """
        coverage_map = np.zeros(self.map_data.shape)
        landmark_coords = np.floor_divide(landmarks[:, :2], resolution).astype('int')
        for l, landmark_cell in enumerate(landmark_coords):
            # Check all cells in range of this landmark and increment the coverage on each one
            for (i, j) in self.range_coverage_map[tuple(landmark_cell[:2])]:
                coverage_map[i, j] += 1
        return coverage_map

    # def get_coverage_map(self, landmarks):
    #     """ Compute the coverage map, indicating how many landmarks are in range of each cell """
    #     coverage_map = np.zeros(self.map_data.shape)
    #     for (i, j) in self.free_cells:
    #         cell_t = np.array([i * resolution, j * resolution, 0.0])
    #         cell_pose = lie.se3(t=cell_t, r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
    #         for l, landmark in enumerate(landmarks):
    #             # Check if landmark is in range
    #             if landmark_detection.distance(cell_pose, landmark) < landmark_detection.MAX_RANGE:
    #                 coverage_map[i, j] += 1
    #     return coverage_map

    def check_coverage(self, landmarks):
        """ Get the coverage map and check if all cells have at least 3 landmarks in range """
        coverage_map = self.get_coverage_map(landmarks)
        for (i, j) in self.free_cells:
            if coverage_map[i, j] < 3:
                return False
        return True

    def valid_configuration(self, landmarks):
        """ Check if the landmark configuration is valid
            - All free cells are covered by at least 3 landmarks.
            - Landmarks are only in landmark-area (non-free).
            - Each non-free cell contains at most one landmark.
        """
        # Get the cell coordinates of the landmarks
        landmark_coords = np.floor_divide(landmarks[:, :2], resolution).astype('int')

        occupied_cells = {}
        for l, landmark_cell in enumerate(landmark_coords):
            # Check if landmark is outside of map boundaries
            if landmark_cell[0] < 0 or landmark_cell[0] >= self.map_data.shape[0] or\
               landmark_cell[1] < 0 or landmark_cell[1] >= self.map_data.shape[1]:
                print("Invalid solution: Landmark {} is out of map range!".format(landmarks[l]))
                return False
            # Check if landmark is in free area
            if self.map_data[landmark_cell[0], landmark_cell[1]] == 255:
                print("Invalid solution: Landmark cell {} is free space!".format(landmark_cell))
                return False
            # Check if landmark is in a cell that is already occupied by another landmark
            if landmark_cell.tobytes() in occupied_cells:
                print("Invalid solution: Landmark cell {} is already occupied!".format(landmark_cell))
                return False
            occupied_cells[landmark_cell.tobytes()] = True
        # Get the coverage map and check if all cells have at least 3 landmarks in range
        if not self.check_coverage(landmarks):
            print("Invalid solution: Not all cells are covered by 3 landmarks!")
            return False
        return True

    def find_cell_max_coverage(self, landmarks):
        """ Find the free cell that maximizes the coverage after placing a new landmark there
            The score of each cell is determined by the amount of landmarks in range:
                0  in range --> 3 points
                1  in range --> 2 points
                2  in range --> 1 points
                >3 in range --> 0 points
            Then, one of the cells with the highest score is randomly selected to place a new landmark
        """
        coverage_map = self.get_coverage_map(landmarks)
        # print("coverage_map:\n{}".format(coverage_map))
        land_score = np.zeros(self.map_data.shape)
        landmark_coords = np.floor_divide(landmarks[:, :2], resolution).astype('int')
        landmark_coords = [tuple(c) for c in landmark_coords]
        for land_cell in self.land_cells:
            # Do not process land cells where we already have a landmark
            land_cell_idx = tuple(land_cell)
            if land_cell_idx in landmark_coords:
                # print("land_cell in landmarks!!!")
                continue
            # Check cells in range of land_cell
            for (i, j) in self.range_coverage_map[land_cell_idx]:
                # Add coverage score to land cell
                if coverage_map[i, j] < 3:
                    land_score[land_cell_idx] += 3 - coverage_map[i, j]

        # print("land_score:\n{}".format(land_score))

        # Select randomly one of the land cells with the highest scores
        max_score = np.max(land_score)
        best_scores = np.argwhere(land_score > int(0.9*max_score))
        # print("best_scores: {}".format(best_scores))
        new_landmark_cell = best_scores[np.random.choice(best_scores.shape[0])]
        print("New landmark cell: {}".format(new_landmark_cell))

        new_landmark = np.array([[new_landmark_cell[0] * resolution, new_landmark_cell[1] * resolution, 0.0]])

        # Create a mask to dislpay the DGOP on top of the map and transpose for displaying
        # landmarks_display = landmarks / resolution
        # plt.imshow(self.map_display, cmap='gray', origin='lower')
        # plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
        # land_score_masked = np.ma.masked_where(self.map_data == 255, land_score).transpose()
        # plt.imshow(land_score_masked, 'viridis', interpolation='none', alpha=1.0, origin='lower')
        # plt.colorbar()
        # plt.show()

        return new_landmark


    def find_landmarks(self):
        """ TODO: Magic """
        landmarks = np.empty((0, 3))
        # Find the free cell that maximizes the coverage
        while not self.check_coverage(landmarks):
            landmarks = np.concatenate((landmarks, self.find_cell_max_coverage(landmarks)), axis=0)
            # print("Landmarks:\n{}".format(landmarks))

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
        # self.find_cell_max_coverage(landmarks)
        return np.array(landmarks)


def main():
    map_data = pgm.read_pgm(map_file)
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(resolution))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width*height))
    print("Map free cells: {}".format(np.count_nonzero(map_data)))

    # Transpose the map data matrix because imshow will treat it
    # as an image and use the first dimension as the Y axis (rows)
    # This is needed only for displaying the map image
    map_display = map_data.transpose()
    # plt.imshow(map_display, cmap='gray', origin='lower')


    # Find a landmark setup that guarantees the desired accuracy
    lpo = LPO(map_data)
    # landmarks = np.array([
    #     [5.0, 40.0, 0.0],
    #     [40.0, 45.0, 0.0],
    #     [55.0, 25.0, 0.0],
    #     [65.0, 55.0, 0.0],
    #     [85.0, 60.0, 0.0],
    #     [100.0, 30.0, 0.0],
    #     [120.0, 60.0, 0.0],
    #     [125.0, 23.0, 0.0],
    # ])

    landmarks = lpo.find_landmarks()
    print("landmarks:\n{}".format(landmarks))


    valid = lpo.valid_configuration(landmarks)
    print("Valid configuration: {}".format(valid))

    # Save the landmarks to a file
    np.save("landmarks", landmarks)

    heatmap = lpo.heatmap_builder.compute_heatmap(landmarks, 'nlls')
    print("heatmap mean: {}".format(np.mean(heatmap)))
    print("heatmap max: {}".format(np.max(heatmap)))
    if np.max(heatmap) > 0.0:
        print("heatmap average: {}".format(np.average(heatmap, weights=(heatmap > 0))))

    # Plot landmarks
    landmarks_display = landmarks / resolution
    plt.imshow(map_display, cmap='gray', origin='lower')
    plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
    # Create a mask to dislpay the heatmap on top of the map and transpose for displaying
    heatmap_masked = np.ma.masked_where(map_data == 0, heatmap).transpose()
    plt.imshow(heatmap_masked, 'viridis', interpolation='none', alpha=1.0, origin='lower')
    plt.colorbar()

    coverage = lpo.get_coverage_map(landmarks)
    print(coverage)

    print("coverage mean: {}".format(np.mean(coverage)))
    print("coverage max: {}".format(np.max(coverage)))
    if np.max(coverage) > 0.0:
        print("coverage average: {}".format(np.average(coverage, weights=(coverage > 0))))

    # Create a mask to dislpay the DGOP on top of the map and transpose for displaying
    plt.figure()
    coverage_masked = np.ma.masked_where(map_data == 0, coverage).transpose()
    plt.imshow(map_display, cmap='gray', origin='lower')
    plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')
    plt.imshow(coverage_masked, 'viridis', interpolation='none', alpha=1.0, origin='lower')
    plt.colorbar()

    plt.show()


if __name__ == '__main__':
    main()
