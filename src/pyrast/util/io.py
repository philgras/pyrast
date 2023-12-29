from pyrast.geometry.scene import Mesh
from typing import Union
from pathlib import Path
import numpy as np


def load_obj(filepath: Union[str, Path]) -> Mesh:
    """
    Simple obj loader. Loads geometry information and stores them into 
    Mesh objects
    Args:
        filepath: path to obj file

    Returns: Mesh object
        
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    line_ary = np.array(lines)
    line_ary = np.char.replace(line_ary, '\n', '')

    vertices = line_ary[np.char.startswith(line_ary, 'v ')]
    vertices = np.array(np.char.split(vertices, ' ').tolist())
    vertices = vertices[:, 1:].astype(float)

    faces = line_ary[np.char.startswith(line_ary, 'f')]
    faces = np.array(np.char.split(faces, ' ').tolist())
    faces = np.array(np.char.split(faces[:, 1:], '/').tolist())
    faces = faces.astype(int) - 1

    vert_uvs = line_ary[np.char.startswith(line_ary, 'vt')]
    vert_uvs = np.array(np.char.split(vert_uvs, ' ').tolist())
    vert_uvs = vert_uvs[:, 1:].astype(float)

    face_uvs = vert_uvs[faces[:, :, 1]]
    faces = faces[:, :, 0]
    return Mesh(
        vertices,
        faces,
        face_attrs={'face_uvs': face_uvs}
    )
