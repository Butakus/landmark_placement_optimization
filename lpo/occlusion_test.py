from time import time

import numpy as np
# Set numpy random seed
# np.random.seed(42)
from matplotlib import pyplot as plt

import pgm
import nlls
import lie_algebra as lie
from landmark_detection import filter_landmarks, filter_landmarks_occlusions

# Resolution = 4
landmarks = np.array([
    [0., 0., 0.],
    [0., 32., 0.],
    [8., 8., 0.],
    [12., 36., 0.],
    [16., 0., 0.],
    [28., 8., 0.],
    [28., 28., 0.],
    [32., 36., 0.],
    [36., 0., 0.],
    [48., 32., 0.],
    [52., 32., 0.],
    [64., 0., 0.],
    [76., 4., 0.],
    [76., 16., 0.],
    [76., 32., 0.],
])

def main(args):
    map_data = pgm.read_pgm(args.map_file)
    resolution = args.map_resolution
    width, height = map_data.shape
    print(map_data)
    print("resolution: {}".format(resolution))
    print("width: {}".format(width))
    print("height: {}".format(height))
    print("Map cells: {}".format(width * height))
    print("Map free cells: {}".format(np.count_nonzero(map_data == 255)))

    # Transpose the map data matrix because imshow will treat it
    # as an image and use the first dimension as the Y axis (rows)
    # This is needed only for displaying the map image
    map_display = map_data.transpose()
    # Cnvert landmarks from meters to pixel coordinates
    landmarks_display = landmarks / resolution
    print("landmarks:\n{}".format(landmarks))
    plt.imshow(map_display, cmap='gray', origin='lower')
    plt.scatter(landmarks_display[:, 0], landmarks_display[:, 1], marker='^', color='m')

    pose = lie.se3(t=[20.0, 4.0, 0.0], r=lie.so3_from_rpy([0.0, 0.0, 0.0]))
    print("origin: {}".format(pose))
    plt.plot(pose[0, 3]/resolution, pose[1, 3]/resolution, 'ro')

    filtered_landmarks = filter_landmarks_occlusions(landmarks, pose, map_data, resolution, 30.0)
    print("filtered_landmarks:\n{}".format(filtered_landmarks))


    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visibility model test')
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
