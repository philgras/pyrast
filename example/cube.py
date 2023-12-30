from pyrast.geometry import *
from pyrast.render import *

import matplotlib.pyplot as plt
import numpy as np

mesh = create_cube()
mesh.textures["diffuse"] = create_checkerboard_texture()
mesh.material = Material()

res = 512
cam = Camera(res, (res, res))
cam.look_at(mesh.location, np.pi / 3, 0.4, distance=4.0)

render = Render(
    rasterizer=Rasterizer(cam.image_size),
    shaders=[
        NormalShader(),
        UVShader(),
        TextureShader(),
        CoordinateShader(),
        PhongShader()
    ]
)

light = DirectionalLight(direction=(0.3, 0., 1.), color=(1., 1., 1.))

out = render(mesh, cam, light)

# collect outputs
mask_img = ~out["mask_img"]
# normal image
normal_img = out["normal_img"] * 0.5 + 0.5
normal_img[mask_img] = 0
# uv image
uv_img = out["uv_img"] * 0.5 + 0.5
uv_img[mask_img] = 0
uv_img = np.concatenate([uv_img, np.zeros_like(uv_img[..., [0]])], axis=-1)
# depth image
depth_img = out["z_buffer"]
shaded_img = out["shaded_img"]
outputs = [normal_img, uv_img, depth_img, shaded_img]

# visualize
_, axes = plt.subplots(1, 4, figsize=(16, 4))
for out, ax in zip(outputs, axes):
    ax.imshow(out)
plt.show()
