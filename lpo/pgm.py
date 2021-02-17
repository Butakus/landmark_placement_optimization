#!/usr/bin/env python3

""" This is a utility module to read PGM image files that represent a map of the environment.
    It can handle ascii (P2) and binary (P5) PGM file formats.
    The image data is converted to a numpy array.
"""

import numpy as np
from matplotlib import pyplot as plt


def read_line(pgm_file):
    """ Read a line from the pgm file.
        Comment lines (#) are skipped and trailing spaces removed.
    """
    while True:
        line = pgm_file.readline()
        if not isinstance(line, str):
            line = line.decode("utf-8")
        if not line.lstrip().startswith('#'):
            return line.rstrip()
        

def check_file_type(pgm_filename):
    """ Check if the pgm file data is ascii (P2) or binary (P5) encoded
        Return the type as a string.
    """
    with open(pgm_filename, 'r') as pgm_file:
        data_type = read_line(pgm_file)
        if data_type != 'P2' and data_type != 'P5':
            raise ValueError("PGM file type must be P2 or P5")
        return data_type

def read_ascii_data(pgm_file, data):
    """ Read the P2 data and fill the numpy array """
    for y in range(data.shape[1]):
        for x, val in enumerate(read_line(pgm_file).split()):
            val = int(val)
            # Invert y coordinate
            y_inv = data.shape[1] - y - 1
            data[x, y_inv] = val

def read_binary_data(pgm_file, data):
    """ Read the P5 data and fill the numpy array """
    for y in range(data.shape[1]):
        for x in range(data.shape[0]):
            val = ord(pgm_file.read(1))
            # Invert y coordinate
            y_inv = data.shape[1] - y - 1
            data[x, y_inv] = val

def read_pgm(pgm_filename):
    """ Read a pgm file and return the data as a numpy array """
    # First, check if file data is ascii (Pw) or binary (P5) encoded
    data_type = check_file_type(pgm_filename)
    file_read_options = 'r' if data_type == 'P2' else 'rb'
    print("PGM data type: {}".format(data_type))

    with open(pgm_filename, file_read_options) as pgm_file:
        # Skip first line (data type)
        read_line(pgm_file)
        # Read data size and depth
        (width, height) = [int(i) for i in read_line(pgm_file).split()]
        depth = int(read_line(pgm_file))
        # TODO: For now only 8bit files are supported.
        # The type of the np.array should change depending on the depth
        assert depth == 255

        print("width: {}".format(width))
        print("height: {}".format(height))
        print("depth: {}".format(depth))

        # Read image data
        data = np.empty((width, height), np.uint8)
        if data_type == 'P2':
            read_ascii_data(pgm_file, data)
        else:
            read_binary_data(pgm_file, data)
        return data


if __name__ == '__main__':
    data = read_pgm("/home/butakus/localization_reference/gazebo/map.pgm")
    print(data)
    plt.imshow(data)
    plt.show()
