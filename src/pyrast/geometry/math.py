from typing import Union
import numpy as np


def transform_points(rigid_transform: np.ndarray, points: np.ndarray):
    """
    Applies the given rigid transform to all points
    Args:
        rigid_transform: rigid transform matrix of shape 4x4
        points: np.array of shape N x 3

    Returns:
        transformed points as np.array of shape N x 4

    """
    N = len(points)
    points_hom = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed = (rigid_transform @ points_hom.T).T
    return transformed[:, :3]


def rotation_vector_to_matrix(rotation_vector: Union[tuple, list, np.ndarray]):
    """
    Turn a axis-angle rotation vector into a 3x3 matrix
    Args:
        rotation_vector : axis-angle vector

    Returns: rotation matrix as np.ndarray of shape 3x3

    """
    # get angle
    rotation_vector = np.array(rotation_vector)
    angle = np.sqrt(np.sum(rotation_vector ** 2))
    if angle == 0:
        return np.eye(3)

    # get rotation axis
    k = rotation_vector / angle

    # compute crossproduct matrix
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], dtype=float)

    # compute rodruigez matrix
    R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle)) * (K @ K)
    return R


def add_rotation_vectors(first: Union[list, tuple, np.ndarray],
                         second: Union[list, tuple, np.ndarray]):
    """
    Adds the rotation described by two axis-angle vectors
    Formulas found here https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
    Args:
        first : first rotation 
        second : second rotation

    Returns:
        axis-angle vector of joined rotation

    """
    first = np.array(first)
    alpha = np.sqrt(np.sum(first ** 2))
    if alpha == 0:
        return second
    l = first / alpha

    second = np.array(second)
    beta = np.sqrt(np.sum(second ** 2))
    if beta == 0:
        return first
    m = second / beta

    sin_a, sin_b = np.sin(alpha/2), np.sin(beta/2)
    cos_a, cos_b = np.cos(alpha/2), np.cos(beta/2)

    # compute angle
    cos_gamma_half = cos_a * cos_b - sin_a * sin_b * np.dot(l, m)
    gamma = np.arccos(cos_gamma_half) * 2

    # compute axis
    sin_gamma_half_n = sin_a * cos_b * l + cos_a * \
        sin_b * m + sin_a * sin_b * np.cross(l, m)
    n = sin_gamma_half_n / np.sin(gamma / 2)
    return n * gamma
