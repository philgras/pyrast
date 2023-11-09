from .scene import Mesh
import numpy as np


def create_checkerboard_texture(repeats=4, dim=256):
    """
    Generate a checkerboard texture
    Args:
        repeats: repitions of the pattern
        dim: size of the texture

    Returns:
        texture of shape dim x dim x 3
        
    """
    pattern = np.array([[0, 1], [1, 0]], dtype=float)
    tex = np.tile(pattern, (repeats, repeats))[..., None]
    tex = np.repeat(tex, 3, axis=-1)
    tex[tex == 0] = 0.5

    H, W = tex.shape[:2]

    u, v = np.meshgrid(np.linspace(0, 1, dim), np.linspace(0, 1, dim))
    uvs = np.stack([u, v]).transpose(1, 2, 0).reshape(-1, 2)
    u, v = uvs[:, 0], uvs[:, 1]

    # transform u,v coords in pixels
    u = np.clip(np.round(u * W - 0.5), 0, W-1).astype(int)
    v = np.clip(np.round(v * H - 0.5), 0, H-1).astype(int)

    colors = tex[v, u]
    return colors.reshape(dim, dim, -1)


def create_cube():
    """
    Creates a unit cube, with uvs and checkerboard texture
    Returns:
       Cube mesh 
    """
    # define a triangle cube
    verts = np.array([
        [1, 1, 1],   # r t f
        [1, -1, 1],   # r b f
        [1, 1, -1],   # r t b
        [1, -1, -1],   # r b b
        [-1, 1, 1],  # l t f
        [-1, -1, 1],  # l b f
        [-1, 1, -1],  # l t b
        [-1, -1, -1],  # l b b
    ], dtype=float)

    faces = np.array([
        [0, 1, 2],   # right
        [2, 1, 3],   # right
        [3, 7, 2],   # back
        [2, 7, 6],   # back
        [6, 7, 5],   # left
        [6, 5, 4],   # left
        [4, 5, 0],   # front
        [0, 5, 1],   # front
        [7, 1, 5],   # bottom
        [7, 3, 1],   # bottom
        [4, 0, 6],   # top
        [6, 0, 2],   # top
    ], dtype=int)

    face_uvs = np.array([
        [[0.25, -0.5], [0.25, 0.], [0.75, -0.5]],     # right
        [[0.75, -0.5], [0.25, 0.], [0.75, 0.]],       # right
        [[0.25, 0.5], [-0.25, 0.5], [0.25, 1]],    # back
        [[0.25, 1], [-0.25, 0.5], [-0.25, 1]],     # back
        [[-0.75, -0.5], [-0.75, 0.], [-0.25, 0.]],   # left
        [[-0.75, -0.5], [-0.25, 0.], [-0.25, -0.5]],  # left
        [[-0.25, -0.5], [-0.25, 0.], [0.25, -0.5]],   # front
        [[0.25, -0.5], [-0.25, 0.], [0.25, 0.]],     # front
        [[-0.25, 0.5], [0.25, 0.], [-0.25, 0.]],     # bottom
        [[-0.25, 0.5], [0.25, 0.5], [0.25, 0.]],     # bottom
        [[-0.25, -0.5], [0.25, -0.5], [-0.25, -1]],  # top
        [[-0.25, -1], [0.25, -0.5], [0.25, -1]]      # top
    ], dtype=float)

    tex = create_checkerboard_texture(repeats=16, dim=256)

    return Mesh(verts, faces, face_attrs={'face_uvs': face_uvs}, texture=tex)
