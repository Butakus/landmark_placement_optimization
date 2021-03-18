# -*- coding: UTF8 -*-
"""
Provides functions for Lie group calculations.
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import typing

import numpy as np
import scipy.spatial.transform as sst


class LieAlgebraException(Exception):
    pass


def hat(v: np.ndarray) -> np.ndarray:
    """
    :param v: 3x1 vector
    :return: 3x3 skew symmetric matrix
    """
    # yapf: disable
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
    # yapf: enable


def vee(m: np.ndarray) -> np.ndarray:
    """
    :param m: 3x3 skew symmetric matrix
    :return: 3x1 vector
    """
    return np.array([-m[1, 2], m[0, 2], -m[0, 1]])


def so3_exp(rotation_vector: np.ndarray):
    """
    Computes an SO(3) matrix from a rotation vector representation.
    :param axis: 3x1 rotation vector (axis * angle)
    :return: SO(3) rotation matrix (matrix exponential of so(3))
    """
    return sst.Rotation.from_rotvec(rotation_vector).as_matrix()


def so3_log(r: np.ndarray, return_angle_only: bool = True,
            return_skew: bool = False) -> typing.Union[float, np.ndarray]:
    """
    :param r: SO(3) rotation matrix
    :param return_angle_only: return only the angle (default)
    :param return_skew: return skew symmetric Lie algebra element
    :return:
        if return_angle_only is False:
            rotation vector (axis * angle)
        or if return_skew is True:
             3x3 skew symmetric logarithmic map in so(3) (Ma, Soatto eq. 2.8)
    """
    if not is_so3(r):
        raise LieAlgebraException("matrix is not a valid SO(3) group element")
    rotation_vector = sst.Rotation.from_matrix(r).as_rotvec()
    angle = np.linalg.norm(rotation_vector)
    if return_angle_only and not return_skew:
        return angle
    if return_skew:
        return hat(rotation_vector)
    else:
        return rotation_vector


def se3(r: np.ndarray = np.eye(3),
        t: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    """
    :param r: SO(3) rotation matrix
    :param t: 3x1 translation vector
    :return: SE(3) transformation matrix
    """
    se3 = np.eye(4)
    se3[:3, :3] = r
    se3[:3, 3] = t
    return se3


def sim3(r: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    """
    :param r: SO(3) rotation matrix
    :param t: 3x1 translation vector
    :param s: positive, non-zero scale factor
    :return: Sim(3) similarity transformation matrix
    """
    sim3 = np.eye(4)
    sim3[:3, :3] = s * r
    sim3[:3, 3] = t
    return sim3


def so3_from_se3(p: np.ndarray) -> np.ndarray:
    """
    :param p: absolute SE(3) pose
    :return: the SO(3) rotation matrix in p
    """
    return p[:3, :3]


def so3_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """
    :param rpy: roll, pitch, yaw angles
    :return: the SO(3) rotation matrix from rpy euler angles
    """
    cos_r = np.cos(rpy[0])
    sin_r = np.sin(rpy[0])
    cos_p = np.cos(rpy[1])
    sin_p = np.sin(rpy[1])
    cos_y = np.cos(rpy[2])
    sin_y = np.sin(rpy[2])
    return np.array([
        [cos_p*cos_y,   -cos_r*sin_y + sin_r*cos_p*cos_y,   sin_r*sin_y + cos_r*sin_p*cos_y],
        [cos_p*sin_y,   cos_r*cos_y + sin_r*sin_p*sin_y,    -sin_r*cos_y + cos_r*sin_p*sin_y],
        [-sin_p,        sin_r*cos_p,                        cos_r*cos_p]
    ])


def se3_inverse(p: np.ndarray) -> np.ndarray:
    """
    :param p: absolute SE(3) pose
    :return: the inverted pose
    """
    r_inv = p[:3, :3].transpose()
    t_inv = -r_inv.dot(p[:3, 3])
    return se3(r_inv, t_inv)


def sim3_inverse(a: np.ndarray) -> np.ndarray:
    """
    :param a: Sim(3) matrix in form:
              s*R  t
               0   1
    :return: inverse Sim(3) matrix
    """
    # det(s*R) = s^3 * det(R)   | det(R) = 1
    # s = det(s*R) ^ 1/3
    s = np.power(np.linalg.det(a[:3, :3]), 1 / 3)
    r = (1 / s * a[:3, :3]).T
    t = -r.dot(1 / s * a[:3, 3])
    return sim3(r, t, 1 / s)


def is_so3(r: np.ndarray) -> bool:
    """
    :param r: a 3x3 matrix
    :return: True if r is in the SO(3) group
    """
    # Check the determinant.
    det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-6)
    # Check if the transpose is the inverse.
    inv_valid = np.allclose(r.transpose().dot(r), np.eye(3), atol=1e-6)
    return det_valid and inv_valid


def is_se3(p: np.ndarray) -> bool:
    """
    :param p: a 4x4 matrix
    :return: True if p is in the SE(3) group
    """
    rot_valid = is_so3(p[:3, :3])
    lower_valid = np.equal(p[3, :], np.array([0.0, 0.0, 0.0, 1.0])).all()
    return rot_valid and lower_valid


def is_sim3(p: np.ndarray, s: float) -> bool:
    """
    :param p: a 4x4 matrix
    :param s: expected scale factor
    :return: True if p is in the Sim(3) group with scale s
    """
    rot = p[:3, :3]
    rot_unscaled = np.multiply(rot, 1.0 / s)
    rot_valid = is_so3(rot_unscaled)
    lower_valid = np.equal(p[3, :], np.array([0.0, 0.0, 0.0, 1.0])).all()
    return rot_valid and lower_valid


def relative_so3(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """
    :param r1, r2: SO(3) matrices
    :return: the relative rotation r1^{⁻1} * r2
    """
    return np.dot(r1.transpose(), r2)


def relative_se3(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    :param p1, p2: SE(3) matrices
    :return: the relative transformation p1^{⁻1} * p2
    """
    return np.dot(se3_inverse(p1), p2)
