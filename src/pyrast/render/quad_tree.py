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


class QuadTree:
    def __init__(self, face_verts, region, max_depth) -> None:
        """
        Constructs a quad tree to spatially separate triangle faces into regions.
        Every leaf node in the tree stores indices that refer to the face_verts array.
        Args:
            face_verts: np.ndarray of shape Fx3x3
            region: region to separate --> defined as tuple (left, top, right, bottom) 
            max_depth: maximum depth of the tree
        """
        bboxes = get_face_bboxes(face_verts)
        face_indices = np.arange(len(bboxes))
        self.root = QuadNode(bboxes, face_indices, region, max_depth)

    def non_empty_regions(self):
        """
        Finds all non-empty leaf nodes in the tree.
        Returns: list containing tuples of (face_idx stored in leaf node, region covered by leaf node)
        """
        return self.root.non_empty_regions()

class QuadNode:
    def __init__(self, boxes, data, region, max_depth):
        """
        Spatial tree node that recursivley splits a 2D region into 4 equally sized
        regions and assigns boxes to the subregions they overlap
        with. each subregion is another quadnode. if max_depth is reached. no subdvision
        takes place and the node represents all faces it was instantiated with.
        Args:
            bboxes: boxes array of shape B x 4 where the last dimension denotes
                    left, top, right, bottom
            data: data associated with each box
            region: region to split --> tuple of 4 float --> (left, top, right, bottom)
            max_depth: maximum depth of the tree from this node
        """
        self.data = []
        self.children = []
        self.region = region

        left, top, right, bottom = self.region

        if max_depth == 0 or len(boxes) == 0:
            self.data = data
            self.is_leaf = True
        else:
            self.is_leaf = False

            pivot_x = (left + right) / 2
            pivot_y = (top + bottom) / 2

            mask_left = boxes[:, 0] < pivot_x
            mask_top = boxes[:, 1] < pivot_y
            mask_right = boxes[:, 2] >= pivot_x
            mask_bottom = boxes[:, 3] >= pivot_y

            top_left = mask_left & mask_top
            top_right = mask_right & mask_top
            bottom_left = mask_left & mask_bottom
            bottom_right = mask_right & mask_bottom

            self.children = [
                QuadNode(boxes[top_left],
                         data[top_left],
                         (left, top, pivot_x, pivot_y),
                         max_depth-1),
                QuadNode(boxes[top_right],
                         data[top_right],
                         (pivot_x, top, right, pivot_y),
                         max_depth-1),
                QuadNode(boxes[bottom_left],
                         data[bottom_left],
                         (left, pivot_y, pivot_x, bottom),
                         max_depth-1),
                QuadNode(boxes[bottom_right],
                         data[bottom_right],
                         (pivot_x, pivot_y, right, bottom),
                         max_depth-1)
            ]

    def __repr__(self):
        return str(self.children)

    def is_in_region(self, point):
        """
        Checks if given 2D point lies inside the region of this node
        Args:
            point (): 
        """
        left, top, right, bottom = self.region
        return left <= point[0] <= right and top <= point[1] <= bottom

    def get_data_at(self, point):
        """
        Returns the data located at child node that contains point
        Args:
            point (): 

        Returns: np.ndarray containing the data of the node
            
        """
        if self.is_leaf:
            assert self.is_in_region(point)
            return self.data
        else:
            data = []
            for child in self.children:
                if child.is_in_region(point):
                    data.append(child.get_data_at(point))
            return np.concatenate(data)


    def non_empty_regions(self):
        """
        Finds all non-empty leaf nodes connected to this node.
        Returns: list containing tuples of (data of leaf node, region of leaf node)
            
        """
        if self.is_leaf:
            if len(self.data) > 0:
                return [[self.data, self.region]]
            else:
                return None
        else:
            res = []
            for child in self.children:
                regions = child.non_empty_regions()
                if regions is not None:
                    res.extend(regions)
            return res
