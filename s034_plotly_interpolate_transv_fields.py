import itertools
import os

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import vofotensors
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from vofotensors.abc import alpha1
from vofotensors.abc import rho1

import mechinterfabric
from mechinterfabric import visualization_plotly
from mechinterfabric.abc import *

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s032")
os.makedirs(directory, exist_ok=True)


converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


#########################################################

almost_zero = 1e-5

index_low, index_medium, index_high = 1, 3, 5

df = pd.DataFrame(
    [
        # ["planar_iso", -2 / 6, 3 / 280, index_low, index_high],
        ["iso", almost_zero, 0, index_medium, index_medium],
        ["ud", 4 / 6, 1 / 35, index_high, index_low],
        ["iso2_max", almost_zero, 1 / 60, index_high, index_high],
        ["iso2_min", almost_zero, -1 / 90, index_low, index_low],
    ],
    columns=["label", "alpha1", "rho1", "index_x", "index_y"],
)


parameterizations = vofotensors.fabric_tensors.N4s_parametric
parameterization = parameterizations["transv_isotropic"]["alpha1_rho1"]
N4_func = sp.lambdify([alpha1, rho1], parameterization)

df["N4"] = df.apply(lambda row: N4_func(alpha1=row["alpha1"], rho1=row["rho1"]), axis=1)

# Q = Rotation.from_rotvec(np.pi / 2 * np.array([0, 1, 0])).as_matrix()
# df = df.set_index("label")
# df.at["planar_iso", "N4"] = con.to_mandel6(
#     np.einsum(
#         "io,jm,kn,lp, omnp->ijkl",
#         Q,
#         Q,
#         Q,
#         Q,
#         con.to_tensor(df.loc["planar_iso"]["N4"]),
#     )
# )
# df.reset_index()

N4s = converter.convert(
    source="mandel6",
    target="tensor",
    quantity="stiffness",
    inp=np.array(df["N4"].to_list()),
)
#########################################################
# Calc entities in df

df_index = df.set_index(["index_x", "index_y"]).index

# New points
indices = [i + 1 for i in range(index_high)]
# indices = [i + 1 for i in range(13) if i in [0, 3, 6, 9, 12]]

indices_points = list(itertools.product(indices, repeat=2))
indices_points = [
    index
    for index in indices_points
    if ((index not in df_index) and (index[0] >= index[1]))
]

new = pd.DataFrame(indices_points, columns=["index_x", "index_y"])

##############################################

import scipy.spatial

# Triangulate
points = df[["index_x", "index_y"]].to_numpy()
tri = scipy.spatial.Delaunay(points)

# Define example targets
# targets = np.array([[2, 3], [3, 4], [4, 5], [10, 10]])
targets = new[["index_x", "index_y"]].to_numpy()

# Plot triangulation
fig = plt.figure()
plt.triplot(points[:, 0], points[:, 1], tri.simplices)

plt.plot(points[:, 0], points[:, 1], "o", label="points")
for index, point in enumerate(points):
    plt.text(*point, str(index))

plt.plot(targets[:, 0], targets[:, 1], "x", label="targets")
for index, point in enumerate(targets):
    plt.text(*point, str(index))

plt.legend()
path_picture = os.path.join(directory, "triangulation" + ".png")
plt.savefig(path_picture, dpi=300)

# Get barycentric coordinates
simplices = tri.find_simplex(targets)

# Make sure all taregts are found
# If a target is not inside of any simplices, -1 is returned by find_simplex()
mask = ~(simplices == -1)
if not mask.all():
    raise Exception("At least on target is outside all triangles")
hidden_triangles = simplices[mask]

dimension = 2

# https://codereview.stackexchange.com/a/41089/255069
transforms_x = tri.transform[hidden_triangles, :dimension]
transforms_y = targets - tri.transform[hidden_triangles, dimension]
tmp = np.einsum("ijk,ik->ij", transforms_x, transforms_y)
bcoords = np.c_[tmp, 1 - tmp.sum(axis=1)]

N4_indices = tri.simplices[hidden_triangles]

##############################################################
# Plot


def do_not_rotate(matrices, weights):
    return np.eye(3)


for interpolation_method in [
    mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_extended_return_values,
]:

    new["N4"] = new.apply(
        lambda row: interpolation_method(
            N4s=N4s[N4_indices[row.name]],
            weights=bcoords[row.name],
            func_interpolation_rotation=do_not_rotate,
        )[0],
        axis=1,
    )

    #############################
    # Plot

    for visualization_method in [
        visualization_plotly.add_N4_plotly,
    ]:

        ############################
        # Set figure

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

        scale = 1.1

        def plot_tp_ensemble(row, text_color=(0, 0, 0)):
            origin = np.array([row["index_x"] * scale, row["index_y"] * scale, 0])

            visualization_method(
                fig=fig,
                N4=np.array(row["N4"]),
                origin=origin,
                nbr_points=100,
                options=None,
                method="fodf",
                limit_scalar=0.55,
            )

            # visualization_method(
            #     fig=fig,
            #     origin=origin,
            #     N4=np.array(row["N4"]),
            # )
            # mlab.text3d(
            #     *(origin + np.array([0, -0.33, 0]) * scale),
            #     text=f"({row['index_x']}, {row['index_y']})",
            #     scale=0.1,
            #     color=text_color,
            # )

        for _, row in new.iterrows():
            plot_tp_ensemble(row=row)

        for _, row in df.iterrows():
            plot_tp_ensemble(row=row, text_color=(1, 0, 0))

        name = "image"

        fig.write_image(os.path.join(directory, name + ".png"))

        # if True:
        #     view = mlab.view()
        #     (azimuth, elevation, distance, focalpoint) = view
        #     mlab.view(*(0, 0, distance, focalpoint))

        # mlab.gcf().scene.parallel_projection = True

        # mlab.orientation_axes()
        # mlab.savefig(filename=os.path.join(directory, "image.png"))

        # mlab.show()


# # Inspect unknown N4s
# tmp = new.set_index(["index_x", "index_y"])
# N4_new = con.to_mandel6(tmp.loc[(1, 2)]["N4"])
# print(np.linalg.eig(N4_new))
# print(np.einsum("i,ji->j", [1, 1, 1, 0, 0, 0], N4_new))
# print(sum(np.einsum("i,ji->j", [1, 1, 1, 0, 0, 0], N4_new)))
