from abc import abstractmethod, ABC
import numpy as np

from pyrast.geometry.scene import Camera, Mesh


def batch_bilinear_interpolate(points, a, b):
    a, b = a[:, None, None], b[:, None]
    p_a = points[:, [0, 2], :] * a + points[:, [1, 3], :] * (1 - a)
    return p_a[:, 0, :] * b + p_a[:, 1, :] * (1 - b)


def batch_nearest_interpolation(points, a, b):
    index = (b > 0.5).astype(int) * 2
    index += (a > 0.5).astype(int)
    return points[np.arange(len(points)), index]


def interpolate_attributes(pix2face, bary_coords, face_attributes):
    """
    face_attributes: F,3,D => for each vertex in face one D-dimensional attribute
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

    def __call__(self, mesh: Mesh, pix2face, barycoords, **kwargs):
        """
        Computes normal vector of mesh per rasterized pixel
        Args:
            pix2face (): 
            barycoords (): 
            mesh: 
            **kwargs: 

        Returns:

        """
        outputs = {}
        vertex_normals = mesh.vertex_normals()
        normal_img = interpolate_attributes(
            pix2face, barycoords, vertex_normals[mesh.faces])
        outputs["vertex_normals"] = vertex_normals
        outputs["normal_img"] = normal_img
        return outputs


class TextureShader(Shader):

    def __init__(self, interpolation_mode="bilinear") -> None:
        super().__init__()
        if interpolation_mode == "bilinear":
            self._interpolate = batch_bilinear_interpolate
        elif interpolation_mode == "nearest":
            self._interpolate = batch_nearest_interpolation
        else:
            raise RuntimeError(
                f"Invalid interpolation mode: {interpolation_mode}")

    def sample_texture(self, mesh, pix2face, bary_coords):
        face_uvs = mesh.face_attrs['face_uvs']
        tex = mesh.texture
        uv_image = interpolate_attributes(pix2face, bary_coords, face_uvs)
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

    def __call__(self, mesh, pix2face, barycoords, **kwargs):
        texture_img = self.sample_texture(mesh, pix2face, barycoords)
        return {"texture_img": texture_img}


class PhongShader(Shader):

    def __call__(self, mesh, cam, light, texture_img, normal_img):
        return super().__call__(*args, **kwargs)
