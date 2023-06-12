#!/usr/bin/env python
""" Core code from https://github.com/shilohc/map2gazebo.
    Modified to directly read a pgm file instead of subscribing to ROS (no ROS used here).
    Also modified to have 3 height values (free, obstacle and land areas)

"""
import cv2
import numpy as np
import trimesh
from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt
import pgm

def show_map(map_data, title="Map debug", show=True):
    map_display = map_data.transpose()
    plt.figure()
    plt.imshow(map_display, cmap='gray', origin='lower')
    plt.title(title)
    if show:
        plt.show()

def debug_contours(map_data, contours, title="Contours", show=True):
    map_display = map_data.copy()
    for c in contours:
        for p in c:
            map_display[p[0, 1], p[0, 0]] = 150
    show_map(map_display, title, show)

def coords_to_loc(coords, resolution):
    x, y = coords
    loc_x = x * resolution
    loc_y = y * resolution
    return np.array([loc_x, loc_y, 0.0])


def get_occupied_regions(map_data, target_value):
    """
    Get occupied regions of map
    """
    # Get a binary image with the target values as 255 and the rest as 0
    bin_map = map_data.copy()
    target_cells = (bin_map == target_value)
    other_cells = (bin_map != target_value)
    bin_map[target_cells] = 255
    bin_map[other_cells] = 0
    print(F"target cells:\n{target_cells}")
    print(F"map_data:\n{map_data}")
    show_map(map_data, "Map data debug", show=False)
    print(F"bin_map:\n{bin_map}")
    show_map(bin_map, "Bin map debug", show=False)
    contours, hierarchy = cv2.findContours(
            bin_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Using cv2.RETR_CCOMP classifies external contours at top level of
    # hierarchy and interior contours at second level.  
    # If the whole space is enclosed by walls RETR_EXTERNAL will exclude
    # all interior obstacles e.g. furniture.
    # https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
    print(F"contours:\n{contours}")
    print(F"hierarchy:\n{hierarchy}")
    hierarchy = hierarchy[0]
    corner_idxs = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    # print(F"corner_idxs:\n{corner_idxs}")
    # print(F"contours:\n{contours}")
    debug_contours(bin_map, [contours[i] for i in corner_idxs], show=False)
    plt.show()
    return [contours[i] for i in corner_idxs]


def contour_to_mesh(contour, height, map_resolution):
    vert_height = np.array([0, 0, height])
    meshes = []
    for point in contour:
        x, y = point[0]
        x = -x # Invert x axis to match the original image
        vertices = []
        new_vertices = [
                coords_to_loc((x, y), map_resolution),
                coords_to_loc((x, y+1), map_resolution),
                coords_to_loc((x+1, y), map_resolution),
                coords_to_loc((x+1, y+1), map_resolution)]
        vertices.extend(new_vertices)
        vertices.extend([v + vert_height for v in new_vertices])
        faces = [[0, 2, 4],
                 [4, 2, 6],
                 [1, 2, 0],
                 [3, 2, 1],
                 [5, 0, 4],
                 [1, 0, 5],
                 [3, 7, 2],
                 [7, 6, 2],
                 [7, 4, 6],
                 [5, 4, 7],
                 [1, 5, 3],
                 [7, 3, 5]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if not mesh.is_volume:
            print("Fixing mesh normals")
            mesh.fix_normals()
        meshes.append(mesh)
    mesh = trimesh.util.concatenate(meshes)
    mesh.remove_duplicate_faces()
    # mesh will still have internal faces.  Would be better to get
    # all duplicate faces and remove both of them, since duplicate faces
    # are guaranteed to be internal faces
    return mesh


def main(args):
    map_data = pgm.read_pgm(args.map_file)

    meshes = []
    # Obstacle contours and meshes
    # obs_contours = get_occupied_regions(map_data, args.obs_value)
    # meshes.extend([contour_to_mesh(c, args.obs_height, args.map_resolution) for c in obs_contours])

    # Landmark area contours and meshes
    land_contours = get_occupied_regions(map_data, args.land_value)
    meshes.extend([contour_to_mesh(c, args.land_height, args.map_resolution) for c in land_contours])

    # Join obstacle and land meshes
    mesh = trimesh.util.concatenate(meshes)

    # Export as STL or DAE
    if args.mesh_type == "stl":
        out_filename = args.map_file.rstrip('.pgm') + '.stl'
        with open(out_filename, 'wb') as f:
            mesh.export(f, "stl")
    elif args.mesh_type == "dae":
        out_filename = args.map_file.rstrip('.pgm') + '.dae'
        with open(out_filename, 'w') as f:
            f.write(trimesh.exchange.dae.export_collada(mesh).decode())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PGM Map to Gazebo world')
    parser.add_argument('map_file', metavar='map_file', type=str,
                        help='Map pgm file')
    parser.add_argument('map_resolution', metavar='map_resolution', type=float,
                        help='Map resolution (m/cell)')
    parser.add_argument('--obs-value', type=int, default=0,
                        help='Map value for obstacle cells')
    parser.add_argument('--land-value', type=int, default=100,
                        help='Map value for landmark area cells')
    parser.add_argument('--obs-height', type=float, default=8.0,
                        help='Height of obstacles')
    parser.add_argument('--land-height', type=float, default=0.1,
                        help='Height of landmark area')
    parser.add_argument('-t', '--mesh_type',
                        default='stl', const='stl',
                        nargs='?', choices=['stl', 'dae'],
                        help='Mesh type (default: %(default)s)')
    args = parser.parse_args()
    print(args)
    main(args)
