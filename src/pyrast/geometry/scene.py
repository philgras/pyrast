from .math import rotation_vector_to_matrix, add_rotation_vectors, transform_points
import numpy as np


class SceneObject:

    def __init__(self, rotation=(0, 0, 0), location=(0, 0, 0), scale=1):
        """
        Initializes scene object with rigid pose
        Args:
            rotation (): rotation as axis angle vector 
            location (): 3D translation
            scale (): uniform scale
        """
        self.R = self.t = self.s = None
        self.set_rigid_transfrom(rotation, location, scale)

    def set_rigid_transfrom(self, rotation=(0, 0, 0), location=(0, 0, 0), scale=1):
        """
        Updates the rigid pose
        Args:
            rotation (): rotation as axis angle vector 
            location (): 3D translation
            scale (): uniform scale
        """
        self.R = np.array(rotation)
        self.t = np.array(location)
        self.s = scale

    def world2object_matrix(self):
        """
        Computes rigid transform from world into object coords.

        Returns:
                 4x4 matrix is in homogeneous coordinates

        """

        # 3x3
        R = rotation_vector_to_matrix(self.R).T
        # 3x1
        t = self.t[None].T
        s = 1 / self.s

        RT = np.concatenate([R, -R @ t], axis=1)
        RT[:3, :3] *= s
        homog = np.array([[0, 0, 0, 1]])
        RT = np.concatenate([RT, homog], axis=0)

        return RT

    def object2world_matrix(self):
        """
        Computes rigid transform from object into world coords.

        Returns:
                 4x4 matrix is in homogeneous coordinates

        """

        # 3x3
        R = rotation_vector_to_matrix(self.R)
        # 3x1
        t = self.t[None].T
        s = self.s

        RT = np.concatenate([R, t], axis=1)
        RT[:3, :3] *= s
        homog = np.array([[0, 0, 0, 1]])
        RT = np.concatenate([RT, homog], axis=0)

        return RT


class Material:
    def __init__(self, ambient=None, diffuse=(0.5, 0.5, 0.5), specular=None,
                 shininess=16):
        self.ambient = np.array(ambient) if ambient is not None else None
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular) if specular is not None else None
        self.shininess = shininess

    # material properties taken from
    # http://learnwebgl.brown37.net/10_surface_properties/surface_properties_color.html
    @staticmethod
    def Brass():
        return Material(
            ambient=(0.329, 0.223, 0.027),
            diffuse=(0.78, 0.568, 0.113),
            specular=(0.992, 0.941, 0.807),
            shininess=27.897,
        )

    @staticmethod
    def Bronze():
        return Material(
            ambient=(0.212, 0.1275, 0.054),
            diffuse=(0.714, 0.428, 0.181),
            specular=(0.3935, 0.271, 0.166),
            shininess=25.6,
        )

    @staticmethod
    def Copper():
        return Material(
            ambient=(0.191, 0.07, 0.0225),
            diffuse=(0.703, 0.270, 0.0828),
            specular=(0.2567, 0.127, 0.086),
            shininess=12.8,
        )

    @staticmethod
    def Chrome():
        return Material(
            ambient=(0.25, 0.25, 0.25),
            diffuse=(0.4, 0.4, 0.4),
            specular=(0.774, 0.774, 0.774),
            shininess=76.8,
        )

    @staticmethod
    def Gold():
        return Material(
            ambient=(0.24, 0.199, 0.07),
            diffuse=(0.7516, 0.606, 0.22),
            specular=(0.628, 0.555, 0.366),
            shininess=51.2,
        )

    @staticmethod
    def PolishedGold():
        return Material(
            ambient=(0.24, 0.22, 0.06),
            diffuse=(0.34, 0.31, 0.09),
            specular=(0.797, 0.723, 0.208),
            shininess=83.2,
        )

    @staticmethod
    def Jade():
        return Material(
            ambient=(0.135, 0.2225, 0.1575),
            diffuse=(0.54, 0.89, 0.063),
            specular=(0.316, 0.316, 0.316),
            shininess=12.8,
        )

    @staticmethod
    def Pearl():
        return Material(
            ambient=(0.25, 0.207, 0.207),
            diffuse=(1.0, 0.829, 0.829),
            specular=(0.296, 0.296, 0.296),
            shininess=11.264,
        )

    @staticmethod
    def BlackRubber():
        return Material(
            ambient=(0.02, 0.02, 0.02),
            diffuse=(0.01, 0.01, 0.01),
            specular=(0.4, 0.4, 0.4),
            shininess=10,
        )

    @staticmethod
    def BlackPlastic():
        return Material(
            ambient=(0.0, 0.0, 0.0),
            diffuse=(0.01, 0.01, 0.01),
            specular=(0.5, 0.5, 0.5),
            shininess=32,
        )


class Mesh(SceneObject):
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_attrs=None, textures=None, material=None):
        """
        Initializes a triangle mesh

        Args:
            vertices: vertices described as array of shape V x 3 
            faces: faces described as array of shape F x 3
            face_attrs: dict containing values of shape F x A_i x ..., where A_i x ... are arbitrary dimensionalities for the i-th entry in the dict. This models face attributes, e.g. uv coordinates.
            material:
        """
        super().__init__()
        self.vertices = vertices
        self.faces = faces
        self.face_attrs = {} if face_attrs is None else face_attrs
        self.textures = {} if textures is None else textures
        self.material = material

    def vertex_normals(self):
        """
        Computes vertex normals
        Returns:
            vertex normals of shape len(vertices) x 3
        """
        face_verts = self.vertices[self.faces]
        vert_normals = np.zeros_like(self.vertices)
        v1 = face_verts[:, 1] - face_verts[:, 0]
        v2 = face_verts[:, 2] - face_verts[:, 0]
        face_normals = np.cross(v1, v2)

        for i in range(3):
            vert_normals[self.faces[:, i]] += face_normals

        vert_normals = vert_normals / \
            np.linalg.norm(vert_normals, axis=1, keepdims=True)

        return vert_normals

    def face_normals(self):
        """
        Computes face normals
        Returns:
            face normals of shape len(face) x 3

        """
        face_verts = self.vertices[self.faces]
        v1 = face_verts[:, 1] - face_verts[:, 0]
        v2 = face_verts[:, 2] - face_verts[:, 0]
        face_normals = np.cross(v1, v2)

        return face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)


class Camera(SceneObject):
    def __init__(self, f, image_size):
        """
        Initializes a camera 
        Args:
            f: focal length
            image_size:  image resolution as tuple (height, width)
        """
        super().__init__()
        h, w = image_size
        self.K = np.array(
            [[f, 0, w / 2],
             [0, f, h / 2],
             [0, 0, 1]], dtype=float)
        self.image_size = image_size

    def look_at(self, center, azi, ele, distance):
        """
        Computes the rigid transform of this camera, such
        that it looks at a point(center) from azimuth, elevation angles on a sphere with radius=distance.

        Args:
            center: center point
            azi: azimuth angle 
            ele: elevation angle 
            distance: sphere radius 
        """
        cam_loc = np.array([np.cos(azi) * np.cos(ele),
                            np.sin(ele),
                            np.sin(azi)*np.cos(ele)]) * distance + center
        azi_rot = (0, -azi + np.pi/2, 0)
        ele_rot = (-ele, 0, 0)
        cam_rot = add_rotation_vectors(azi_rot, ele_rot)
        self.set_rigid_transfrom(location=cam_loc, rotation=cam_rot)

    def world2view_matrix(self):
        """
        Computes the rigid transformation from world to view coordinates, e.g. world2object and Y & Z are flipped
        Returns:
            4x4 matrix
        """

        RT = super().world2object_matrix()
        R = np.eye(4)
        R[:3, :3] = rotation_vector_to_matrix(np.array([np.pi, 0, 0]))
        return R @ RT

    def project_mesh(self, mesh: Mesh):
        """
        Projects triangle mesh onto image plane of this camera

        Args:
            mesh (): 

        Returns:

        """
        to_world = mesh.object2world_matrix()
        vertices = transform_points(to_world, mesh.vertices)
        return self.project_points(vertices)

    def project_points(self, points):
        """
        Points must be in world coordinates
        """
        to_cam = self.world2view_matrix()

        # perform rigid transform
        cam_verts = transform_points(to_cam, points)

        # project
        proj_verts = self.K @ cam_verts.T

        u = proj_verts[0] / proj_verts[2]
        v = proj_verts[1] / proj_verts[2]
        z = proj_verts[2]

        return np.vstack([u, v, z]).T


class DirectionalLight(SceneObject):

    def __init__(self,
                 direction=(1., 0., 0.),
                 color=(1.0, 1.0, 1.0),
                 ambient_intensity=0.1,
                 diffuse_intensity=0.7,
                 specular_intensity=0.2
                 ) -> None:

        self.direction = np.array(direction)
        self.color = np.array(color)
        self.ambient_intensity = ambient_intensity * self.color
        self.diffuse_intensity = diffuse_intensity * self.color
        self.specular_intensity = specular_intensity * self.color

    def normalized_direction(self):
        norm = np.linalg.norm(self.direction)
        assert norm > 1e-6
        return self.direction / norm
