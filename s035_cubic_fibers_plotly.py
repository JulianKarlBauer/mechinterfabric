import itertools
import operator

import mechkit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mechinterfabric.visualization_plotly


np.set_printoptions(linewidth=100000, precision=5)


def evenly_distributed_vectors_on_sphere(nbr_vectors=1000):
    """
    Define nbr_vectors evenly distributed vectors on a sphere
    Using the golden spiral method kindly provided by
    stackoverflow-user "CR Drost"
    https://stackoverflow.com/a/44164075/8935243
    """
    from numpy import pi, cos, sin, arccos, arange

    indices = arange(0, nbr_vectors, dtype=float) + 0.5

    phi = arccos(1 - 2 * indices / nbr_vectors)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    orientations = np.column_stack((x, y, z))
    return orientations


con = mechkit.notation.Converter()

cubic_transformations = [
    np.array(tup)
    for tup in [
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
        ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
        ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
        ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
        ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, -1), (-1, 0, 0)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ]
]


def cast_and_normalize(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)


def make_cubic(fibers):
    workload = list(zip(*list(itertools.product(cubic_transformations, fibers))))
    cubic_fibers = list(map(operator.matmul, *workload))
    return cubic_fibers


def analyse(tensor):
    analysis = mechinterfabric.FOT4Analysis(tensor).analyse()


###############################################

vec_min = cast_and_normalize([1, 0, 0])
vec_max = cast_and_normalize([1, 1, 1])

factors = [1 / 3, 1 / 2, 2 / 3]  # np.linspace(0.1, 0.9, 5)

experiments = {
    "min": [
        (1, 0, 0),
    ],
    "iso": evenly_distributed_vectors_on_sphere(100),
    "max": [
        (1, 1, 1),
    ],
    **{
        f"linear_{factor:.2f}": [vec_min + factor * (vec_max - vec_min)]
        for factor in factors
    },
}

# Cast
experiments = {
    key: [cast_and_normalize(fiber) for fiber in fibers]
    for key, fibers in experiments.items()
}

# Make cubic
experiments = {key: make_cubic(fibers=fibers) for key, fibers in experiments.items()}


# Plot

fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"is_3d": True}]],
    subplot_titles=[
        f"title",
    ],
)
fig.update_layout(scene_aspectmode="data")

fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, visible=False),
        yaxis=dict(showticklabels=False, visible=False),
        zaxis=dict(showticklabels=False, visible=False),
        camera=dict(projection=dict(type="orthographic")),
    )
)

origin = [0, 0, 0]

rotation = cubic_transformations[0]


def add_pseudo_cylinder(fig, origin, rotation, nbr_points=300, ratio=20):

    limit = 0.5 * ratio

    vectors = mechinterfabric.visualization.get_unit_vectors(nbr_points=nbr_points)
    vectors[0, ...] = vectors[0, ...] * ratio
    vectors[0, ...] = np.clip(vectors[0, ...], -limit, limit)

    xyz = mechinterfabric.visualization.shift_b_origin(xyz=vectors, origin=origin)

    xyz = np.einsum("ji, i...->j...", rotation, xyz)

    surface = go.Surface(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        # surfacecolor=surfacecolor,
        showscale=False,
        colorscale=[
            [0, "rgb(0.2, 0.7, 0.2)"],
            [1, "rgb(0.2, 0.7, 0.2)"],
        ],
        # **mechinterfabric.visualization_plotly.get_default_options(),
    )

    fig.add_trace(surface)


add_pseudo_cylinder(fig=fig, origin=origin, rotation=rotation)

# import pygmsh
# import numpy as np
# import os

# with pygmsh.occ.Geometry() as geom:
#     geom.add_cylinder(x0=(0,0,0), axis=[1,1,1], radius=1)
#     mesh = geom.generate_mesh()
