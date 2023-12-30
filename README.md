# pyrast
A small library for 3D rendering in Python soley based on numpy.
It provides APIs to model scene geometry (meshes, lights, materials, cameras), perform rasterization, shading, texture sampling and rudimentary .obj file loading and timing.
As the implementation is purely numpy-based, the rendering is not accelerated by a GPU and in consequence not real-time.
![out](https://github.com/philgras/pyrast/assets/13590652/00fbb51b-d6a8-420e-a515-447b0ed65c06)

# Installation
pyrast can be installed with pip as every other package. Numpy is the only dependency.
To run the examples, matplotlib is also required for visualization.

```sh
git clone https://github.com/philgras/pyrast.git
cd pyrast
pip install .
# or if you would like to try the examples
pip install .[dev]
```

# Usage
A short example on how to use the basic API is provided below. Here, a mesh is loaded from an .obj-file. Material, light, camera and renderer are also defined.
In the last line, the scenes gets rasterized and the result is processed by a sequence of shaders.

```python
from pyrast.geometry.scene import Camera, DirectionalLight, Material
from pyrast.render import *
from pyrast.util.io import load_obj

import numpy as np

# load mesh and set material
mesh = load_obj("./path/to/your/mesh.obj")
material = Material.Pearl()
mesh.material = material

# setup camera
res = 512
cam = Camera(res, (res, res))
cam.look_at(mesh.location, np.pi / 3, 0.4, distance=4.0)


# setup light
light = DirectionalLight(direction=(0.3, 0., 1.), color=(1., 1., 1.))

# define render
render = Render(
    rasterizer=Rasterizer(cam.image_size),
    # setup the shaders you would like to run in sequence
    # shaders can use the outputs of their predecessors
    shaders=[
        NormalShader(),
        CoordinateShader(),
        PhongShader()
    ]
)

# out is a dictionary containing shader and rasterization outputs
out = render(mesh, cam, light)
# access the output of different shaders
# to see what outputs every shader and the renderer itself produces,
# look into src/pyrast/render/render.py
normal_img = out["normal_img"]
shaded_img = out["shaded_img"]
depth_img = out["z_buffer"]

```

More examples are in the `./examples` folder.

# Disclaimer
This library was written as part of my freetime project to learn more about rasterization.
Throughout the project, I found some insightful material on [learnopengl](https://learnopengl.com/) and [scratchapixel](https://www.scratchapixel.com/lessons/3d-basic-rendering/rasterization-practical-implementation/overview-rasterization-algorithm.html).
In case, you would also want to learn about the topic, I can recommend both.
They are really helpful. 

If this library is of any use for you and you wish to have additional features, feel free to submit an issue or a pull request.  
