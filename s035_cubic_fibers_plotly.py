import itertools
import operator
from fractions import Fraction

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
    # "iso": evenly_distributed_vectors_on_sphere(100),
    **{
        f"linear_{factor:.2f}": [vec_min + factor * (vec_max - vec_min)]
        for factor in factors
    },
    "max": [
        (1, 1, 1),
    ],
}

# Cast
experiments = {
    key: [cast_and_normalize(fiber) for fiber in fibers]
    for key, fibers in experiments.items()
}

# Make cubic
experiments = {key: make_cubic(fibers=fibers) for key, fibers in experiments.items()}


# Ingest tensor
ingest = {}
for name, fibers in experiments.items():
    fot4 = mechkit.fabric_tensors.first_kind_discrete(orientations=fibers, order=4)
    fot4_mandel = con.to_mandel6(fot4)
    dev4 = mechkit.operators.dev(fot4)
    dev4_mandel = con.to_mandel6(dev4)

    ingest[name] = {
        "fot4_mandel": fot4_mandel,
        "dev4_mandel": dev4_mandel,
        "d_1": dev4_mandel[0, 1],
    }

    analysis = mechinterfabric.FOT4Analysis(fot4_mandel).analyse()
    ingest[name].update(
        {
            "analysis": analysis,
        }
    )

d1s = [ing["d_1"] for name, ing in ingest.items()]


d1s_fracs = [Fraction(val).limit_denominator(1000) for val in d1s]
d1s_strings = [f"{frac.numerator}/{frac.denominator}" for frac in d1s_fracs]

#########################################


offset = np.array([1, 0, 0])

origins = [offset * d1 * 900 for d1 in d1s]

#########################################

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


def add_pseudo_cylinder(fig, origin, rotation, nbr_points=50, ratio=60, limit=10):

    # limit = 0.5 * ratio

    vectors = mechinterfabric.visualization.get_unit_vectors(nbr_points=nbr_points)
    vectors[0, ...] = vectors[0, ...] * ratio
    vectors[0, ...] = np.clip(vectors[0, ...], -limit, limit)

    xyz = np.einsum("ji, i...->j...", rotation, vectors)

    xyz = mechinterfabric.visualization.shift_b_origin(xyz=xyz, origin=origin)

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


for index, (key, fibers) in enumerate(experiments.items()):

    origin = origins[index]

    for fiber in fibers:
        rotation = mechkit.material.TransversalIsotropic._get_rotation_matrix(
            None, start_vector=[1, 0, 0], end_vector=fiber
        )

        add_pseudo_cylinder(fig=fig, origin=origin, rotation=rotation)


ones = np.ones_like(d1s)

# fig.update_traces(mode="lines+markers+text")

scale_z = 14

color = "black"  # "#5D69B1"

textfont = dict(
    family="Courier New, monospace",
    size=26,
    color="black",
)

# Plot line
fig.add_trace(
    go.Scatter3d(
        x=[origin[0] for origin in origins],
        y=ones * 0,
        z=-ones * scale_z,
        text=d1s_strings,
        textposition="bottom center",
        textfont=textfont,
        hovertext=None,
        mode="lines+markers+text",
        marker=dict(color=color, size=6),
        line=dict(color=color, width=3),
    )
)

fig.update_layout(
    scene=dict(
        annotations=[
            dict(
                x=origins[-1][0] + 10,
                y=0,
                z=-scale_z,
                text="$d_1$",  # r"\resizebox{10}{!}{$d_1$}",  # "$d_1$",
                textangle=0,
                font=dict(
                    family="Courier New, monospace",
                    # size=54,
                    color="black",
                ),
                # ax=60,
                # ay=0,
                # az=0,
                showarrow=False,
                # arrowcolor="#5D69B1",
                # arrowsize=3,
                # arrowwidth=3,
                # arrowhead=1,
                # xanchor="left",
                # yanchor="bottom",
            ),
        ],
    ),
)
