import numpy as np
from .quad_tree import QuadNode, get_face_bboxes


def edge_function(v0, v1, v2):
    """
    Computes edge function with broadcasting enabled
    v0, v1, v2 are supposed to be of either shape 1xNx2 or Mx1x2
    result will be MxN
    """
    p1 = v1[:, :, :2] - v0[:, :, :2]
    p2 = v2[:, :, :2] - v0[:, :, :2]
    return p1[:, :, 0] * p2[:, :, 1] - p1[:, :, 1] * p2[:, :, 0]


def rasterize_region(face_verts, pixels, perspective_correct=True):
    # P x 3, P x 1, P x 1
    # face verts of shape Fx3x3
    # individual vectors of shape Fx3
    v0, v1, v2 = face_verts[:, None, 0], face_verts[:,
                                                    None, 1], face_verts[:, None, 2]

    # pixels of shape Px2
    p = pixels[None]

    # output of edge function has shape FxP
    e_02 = edge_function(v0, p, v2)
    e_01 = edge_function(v0, v1, p)
    e_12 = edge_function(v1, v2, p)

    # area has shape Fx1
    area = edge_function(v0, v1, v2)

    # lambdas of shape FxP
    lambda_0 = e_12 / area
    lambda_1 = e_02 / area
    lambda_2 = e_01 / area

    if perspective_correct:
        # derivation -> p1' in screen space, p1 in cam space, lambda -> l
        # p0' l0' + p1' l1' + p2' l2' = (1 / z) * (p0 l0 + p1 l1 + p2 l2)
        #                             = p0' (l0 z0 / z) + p1' (l1 z1 / z) + p2' ((1 - l1 - l2) z2 / z)
        #
        # gives three eqations with three unknowns
        #
        # l0' = (l0 z0 / z)
        # l1' = (l1 z1 / z)
        # l2' = ((1 - l1 - l2) z2 / z)
        #
        # solving for l0, l1, z yields
        #
        # l0 = l0' z0 / z
        # l1 = l1' z1 / z
        # 1 / z = (1 / z0) * l0' + (1 / z1) * l1' + (1 / z2) * l2'
        #
        # l2 = l2' z2 / z by substituting back in

        z0, z1, z2 = 1 / v0[..., 2], 1 / v1[..., 2], 1 / v2[..., 2]
        # z of shape FxP
        z = 1 / (lambda_0 * z0 + lambda_1 * z1 + lambda_2 * z2)
        lambda_0 = lambda_0 * z * z0
        lambda_1 = lambda_1 * z * z1
        lambda_2 = lambda_2 * z * z2
    else:
        z0, z1, z2 = v0[..., 2], v1[..., 2], v2[..., 2]
        z = lambda_0 * z0 + lambda_1 * z1 + lambda_2 * z2

    # filter by visibility and depth
    is_visible = (e_01 <= 0) & (e_02 <= 0) & (e_12 <= 0)
    has_no_visible = is_visible.sum(axis=0) == 0

    # set z of invisible face to inf
    z[~is_visible] = np.inf

    # get closest face
    closest_face_idx = np.argmin(z, axis=0)
    pixel_idx = np.arange(len(closest_face_idx))

    # set depth
    depth = z[closest_face_idx, pixel_idx]

    # set barycentric coordinates
    bary_coords = np.stack([
        lambda_0[closest_face_idx, pixel_idx],
        lambda_1[closest_face_idx, pixel_idx],
        lambda_2[closest_face_idx, pixel_idx]
    ]).T

    # set invalid values
    depth[has_no_visible] = -1
    bary_coords[has_no_visible] = -1
    closest_face_idx[has_no_visible] = -1

    return bary_coords, depth, closest_face_idx


def rasterize_tree(proj_verts, faces, image_size, tree_depth=4, perspective_correct=True):
    height, width = image_size
    face_verts = proj_verts[faces]

    # create quad tree to only rasterize regions
    # where triangles are located
    bboxes = get_face_bboxes(face_verts)
    face_indices = np.arange(len(bboxes))
    quad_tree = QuadNode(tree_depth, face_indices, bboxes, 0, 0, height, width)
    regions = quad_tree.non_empty_regions()

    pix2face = np.zeros(image_size, dtype=int) - 1
    bary_coords = np.zeros((height, width, 3)) - 1
    z_buffer = np.zeros(image_size) - 1

    for face_indices, region_bbox in regions:
        l, t, r, b = [int(v) for v in region_bbox]
        y, x = np.meshgrid(range(int(t), int(b)), range(int(l), int(r)))
        pixels = np.stack([x.reshape(-1), y.reshape(-1)]).T

        face_verts_i = face_verts[face_indices]
        bary_coords_i, z, closest_face = rasterize_region(
            face_verts_i.astype(np.float32),
            pixels.astype(np.float32),
            perspective_correct
        )

        # correct face indices
        mask = closest_face > -1
        closest_face[mask] = face_indices[closest_face[mask]]

        h, w = int(b-t), int(r-l)
        pix2face[t:b, l:r] = closest_face.reshape(h, w).T
        z_buffer[t:b, l:r] = z.reshape(h, w).T
        bary_coords[t:b, l:r] = bary_coords_i.reshape(
            h, w, 3).transpose((1, 0, 2))

    return pix2face, bary_coords, z_buffer


class Rasterizer:
    def __init__(self, image_size, perspective_correct=True):
        self.image_size = image_size
        self.perspective_correct = perspective_correct

    def __call__(self, vertices, faces, tree_depth=4):
        # transform into normalized space
        return rasterize_tree(vertices, faces, 
                              self.image_size, tree_depth, self.perspective_correct)
