import numpy as np


def get_face_bboxes(face_verts):
    """
    Computes bouding boxes for each projected face
    Args:
        face_verts : F x 3 x 3 tensor containing face vertex positions in screen space
    Returns: bbox tensor of shape F x 4
        
    """
    min_x, min_y = np.min(face_verts, axis=1)[..., :2].T
    max_x, max_y = np.max(face_verts, axis=1)[..., :2].T
    return np.vstack([min_x, min_y, max_x, max_y]).T


class QuadNode:

    def __init__(self, max_depth, faces, bboxes, left, top, right, bottom):
        if max_depth == 0:
            self.children = faces
            self.is_leaf = True
            self.bbox = [left, top, right, bottom]
        elif len(faces) == 0:
            self.children = []
            self.is_leaf = True
        else:
            self.is_leaf = False

            pivot_x = (left+right) / 2
            pivot_y = (top+bottom) / 2
            self.split = pivot_x, pivot_y

            mask_left = bboxes[:, 0] < pivot_x
            mask_top = bboxes[:, 1] < pivot_y
            mask_right = bboxes[:, 2] >= pivot_x
            mask_bottom = bboxes[:, 3] >= pivot_y

            top_left = mask_left & mask_top
            top_right = mask_right & mask_top
            bottom_left = mask_left & mask_bottom
            bottom_right = mask_right & mask_bottom

            self.children = [
                [QuadNode(max_depth-1,
                          faces[top_left],
                          bboxes[top_left],
                          left, top, pivot_x, pivot_y),
                 QuadNode(max_depth-1,
                          faces[top_right],
                          bboxes[top_right],
                          pivot_x, top, right, pivot_y)],
                [QuadNode(max_depth-1,
                          faces[bottom_left],
                          bboxes[bottom_left],
                          left, pivot_y, pivot_x, bottom),
                 QuadNode(max_depth-1,
                          faces[bottom_right],
                          bboxes[bottom_right],
                          pivot_x, pivot_y, right, bottom)]
            ]

    def __repr__(self):
        return str(self.children)

    def get_faces_at(self, point):
        if self.is_leaf:
            return self.children
        else:
            index_y = 0 if point[1] < self.split[1] else 1
            index_x = 0 if point[0] < self.split[0] else 1
            return self.children[index_y][index_x].get_faces_at(point)

    def non_empty_regions(self):
        if self.is_leaf:
            if len(self.children) > 0:
                return [[self.children, self.bbox]]
            else:
                return None
        else:
            res = []
            for child_group in self.children:
                for child in child_group:
                    regions = child.non_empty_regions()
                    if regions is not None:
                        res.extend(regions)
            return res
