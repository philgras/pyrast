from pyrast.geometry.scene import *
from abc import abstractmethod, ABC
import numpy as np


def batch_bilinear_interpolate(points, a, b):
    """
    Bilinear interpolation between points according to weights a, b
    Args:
        points: array of shape Bx4xD --> second dimension contains
            (top, left), (top, right), (bottom, left), (bottom, right) points to be
            interpolated
        a: interplation weight from left to right of shape B
        b: interpolation weight from top to bottom of shape B

    Returns: Interpolated points of shape BxD

    """
    a, b = a[:, None, None], b[:, None]
    p_a = points[:, [0, 2], :] * a + points[:, [1, 3], :] * (1 - a)
    return p_a[:, 0, :] * b + p_a[:, 1, :] * (1 - b)


def batch_nearest_interpolation(points, a, b):
    """
    Nearest-neighbor interpolation between points according to weights a, b
    Args:
        points: array of shape Bx4xD --> second dimension contains
            (top, left), (top, right), (bottom, left), (bottom, right) points to be
            interpolated
        a: interplation weight from left to right of shape B
        b: interpolation weight from top to bottom of shape B

    Returns: Interpolated points of shape BxD

    """
    index = (b > 0.5).astype(int) * 2
    index += (a > 0.5).astype(int)
    return points[np.arange(len(points)), index]


def interpolate_attributes(pix2face, bary_coords, face_attributes):
    """
    Interpolates vertex attributes across triangle face using barycentric
    coordinates.

    Args:
        pix2face: face_idx output of the rasterizer of shape HxW
        bary_coords: barycentric coordinates output of the rasterizer of shape HxWx3
        face_attributes: vertex attributes of dimensione D grouped for every
                         triangle face --> shape is Fx3xD

    Returns:
        array of shape HxWxD
    """

    H, W = pix2face.shape
    D = face_attributes.shape[-1]
    res = np.zeros((H, W, D))

    mask = pix2face > -1

    # shape M x 3 x D
    pix_attributes = face_attributes[pix2face[mask]]

    # shape M x 3 x 1
    bary_coords = bary_coords[mask][..., None]

    # interpolate --> shape M x D
    res[mask] = (pix_attributes * bary_coords).sum(axis=1)
    return res


class Render:

    def __init__(self, rasterizer, shaders=[]) -> None:
        self.rasterizer = rasterizer
        self.shaders = shaders

    def __call__(self, mesh, cam, light=None):
        # project mesh and make resolutions match
        projected_verts = cam.project_mesh(mesh)
        w_ratio = self.rasterizer.image_size[1] / cam.image_size[1]
        h_ratio = self.rasterizer.image_size[0] / cam.image_size[0]
        projected_verts[:, 0] *= w_ratio
        projected_verts[:, 1] *= h_ratio

        # rasterize
        pix2face, barycoords, z_buffer = self.rasterizer(
            projected_verts, mesh.faces
        )

        outputs = {
            "barycoords": barycoords,
            "z_buffer": z_buffer,
            "pix2face": pix2face,
            "mask_img": pix2face > -1,
            "mesh": mesh,
            "cam": cam,
            "light": light,
            "projected_verts": projected_verts
        }

        # run all shaders
        for shader in self.shaders:
            outputs.update(shader(**outputs))

        return outputs


class Shader(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class NormalShader(Shader):
    def __init__(self, use_vertex_normals=True):
        """

        Args:
            use_vertex_normals (): 
        """
        self._use_vertex_normals = use_vertex_normals

    def __call__(self, mesh: Mesh, pix2face,
                 barycoords, mask_img, **kwargs):
        """
        Per rasterized pixel the normal of the respective surface point
        is computed.
        """
        outputs = {}
        if self._use_vertex_normals:
            vertex_normals = mesh.vertex_normals()
            normal_img = interpolate_attributes(
                pix2face,
                barycoords,
                vertex_normals[mesh.faces]
            )
            outputs["vertex_normals"] = vertex_normals
        else:
            face_normals = mesh.face_normals()
            normal_img = np.zeros_like(barycoords)
            normal_img[mask_img] = face_normals[pix2face[mask_img]]
            outputs["face_normals"] = face_normals
        outputs["normal_img"] = normal_img
        return outputs


class CoordinateShader(Shader):
    def __call__(self, mesh, pix2face, barycoords, mask_img, **kwargs):
        """
        Per pixel, the 3D coordinate of the corresponding surface point is 
        computed.
        """
        out = np.zeros_like(barycoords)
        mask = mask_img
        face_idx = pix2face[mask]
        faces = mesh.faces[face_idx]
        face_verts = mesh.vertices[faces]
        points = face_verts * barycoords[mask][..., None]
        points = points.sum(axis=-2)
        out[mask] = points
        return {"coordinate_img": out}


class UVShader(Shader):
    def __call__(self, mesh, pix2face, barycoords, **kwargs):
        """
        Per pixel, the uv coordinate of the corresponding surface point is 
        computed.
        """
        face_uvs = mesh.face_attrs['face_uvs']
        uv_image = interpolate_attributes(pix2face, barycoords, face_uvs)
        return {"uv_img": uv_image}


class TextureShader(Shader):

    def __init__(self, texture_names=None, interpolation_mode="bilinear") -> None:
        super().__init__()
        self.texture_names = texture_names
        if interpolation_mode == "bilinear":
            self._interpolate = batch_bilinear_interpolate
        elif interpolation_mode == "nearest":
            self._interpolate = batch_nearest_interpolation
        else:
            raise RuntimeError(
                f"Invalid interpolation mode: {interpolation_mode}")

    def sample_texture(self, tex, uv_image, pix2face):
        mask = pix2face > -1

        img_H, img_W = pix2face.shape[:2]
        tex_H, tex_W, C = tex.shape
        res = np.zeros((img_H, img_W, C))
        uvs = uv_image[mask]
        u, v = uvs[:, 0], uvs[:, 1]

        # transform u,v coords in pixels
        u = (u + 1) / 2 * (tex_W - 1)
        v = (v + 1) / 2 * (tex_H - 1)

        # get corner points
        u_low, u_high = np.floor(u).astype(int), np.ceil(u).astype(int)
        v_low, v_high = np.floor(v).astype(int), np.ceil(v).astype(int)

        # interpolation weights
        a, b = 1 - (u - u_low), 1 - (v - v_low)

        # interpolate corner points
        points = tex[v_low, u_low], tex[v_low,
                                        u_high], tex[v_high, u_low], tex[v_high, u_high]
        points = np.stack(points).transpose(1, 0, 2)
        colors = self._interpolate(points, a, b)
        res[mask] = colors
        return res

    def __call__(self, mesh, uv_img, pix2face, **kwargs):
        """
        Samples textures of the mesh for every rasterized suface point.

        """
        if self.texture_names is None:
            names = list(mesh.textures.keys())
        else:
            names = self.texture_names

        out = {}
        for name in names:
            texture = mesh.textures[name]
            texture_img = self.sample_texture(texture, uv_img, pix2face)
            out[name] = texture_img
        return out


class PhongShader(Shader):
    def __call__(self, mesh: Mesh, cam: Camera, light: DirectionalLight,
                 normal_img, coordinate_img, mask_img, **kwargs):
        """
        Computes phong shading per pixel.  
        """
        material = mesh.material

        # set the material colors
        # diffuse
        if "diffuse" in kwargs:
            diffuse = kwargs["diffuse"][mask_img]
        else:
            diffuse = material.diffuse[None]

        # ambient
        if material.ambient is None:
            ambient = diffuse
        else:
            ambient = material.ambient[None]

        # specular
        if material.specular is None:
            specular = light.color[None]
        else:
            specular = material.specular[None]

        shininess = material.shininess
        normal = normal_img[mask_img]

        # ambient
        out = np.zeros_like(normal_img)
        out[mask_img] = light.ambient_intensity[None] * ambient

        # diffuse
        light_direction = light.normalized_direction()[None]
        LN = np.sum(normal * light_direction, axis=-1)
        factor = np.clip(LN, 0, None)
        out[mask_img] += light.diffuse_intensity[None] * \
            diffuse * factor[..., None]

        # specular
        # compute view direction
        points = coordinate_img[mask_img]
        view_direction = cam.location - points
        norm = np.linalg.norm(view_direction, axis=-1)
        assert np.all(norm > 1e-5)
        view_direction = view_direction / norm[..., None]

        # compute reflect direction
        reflect_direction = 2 * LN[..., None] * normal - light_direction
        norm = np.linalg.norm(reflect_direction, axis=-1)
        assert np.all(norm > 1e-5)
        reflect_direction = reflect_direction / norm[..., None]

        # compute specular
        backface_mask = LN == 0
        VR = np.sum(view_direction * reflect_direction,
                        axis=-1)
        factor = np.clip(VR, 0, None) ** shininess
        factor[backface_mask] = 0
        normalizer = (shininess + 2) / 2 * np.pi

        out[mask_img] += normalizer * light.specular_intensity[None] * \
            specular * factor[..., None]

        return {"shaded_img": np.clip(out, 0., 1.)}
