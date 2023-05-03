import itertools
import os

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

import mechinterfabric
from mechinterfabric import visualization_plotly

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s037")
os.makedirs(directory, exist_ok=True)


converter = mechkit.notation.ExplicitConverter()


#########################################################


def N2_from_row(row):
    return [
        [row["n11"], row["n12"], row["n13"]],
        [row["n12"], row["n22"], row["n23"]],
        [row["n13"], row["n23"], row["n33"]],
    ]


def N4_from_row(row):
    # N4 is completely index-symmetric
    N4 = np.zeros((3, 3, 3, 3), dtype=np.float64)

    columns = [
        "3333",
        "3332",
        "3322",
        "3222",
        "2222",
        "3331",
        "3321",
        "3221",
        "2221",
        "3311",
        "3211",
        "2211",
        "3111",
        "2111",
        "1111",
    ]

    for column_key in columns:
        permutations = itertools.permutations([int(item) - 1 for item in column_key])
        for perm in permutations:
            N4[perm] = row[column_key]
    return N4.tolist()


#########################################################
# Read N2
df_N2 = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_01_31_1124_N2.csv"),
    header=0,
    sep=",",
)
df_N2.columns = df_N2.columns.str.strip()

# Read N4
df_N4 = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_01_31_1124_N4.csv"),
    header=0,
    sep=",",
)
df_N4.columns = df_N4.columns.str.strip()

# Merge
df = df_N2.merge(df_N4)


df["N2"] = df.apply(N2_from_row, axis=1)
df["N4"] = df.apply(N4_from_row, axis=1)

N2s = np.array(df["N2"].to_list())
N4s = np.array(df["N4"].to_list())


#########################################################
# Calc entities in df

df_index = df.set_index(["index_x", "index_y"]).index

N4s_df = np.array(df["N4"].to_list())


# New points
indices = [i + 1 for i in range(13)]
# indices = [i + 1 for i in range(13) if i in [0, 3, 6, 9, 12]]

indices_points = list(itertools.product(indices, repeat=2))
indices_points = [index for index in indices_points if index not in df_index]

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


for interpolation_method in [
    mechinterfabric.interpolation.interpolate_N4_decomp_extended_return_values,
    # mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_extended_return_values,
    # mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_analysis_extended_return_values,
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

    new["N4"] = new.apply(
        lambda row: interpolation_method(
            N4s=N4s[N4_indices[row.name]],
            weights=bcoords[row.name],
            func_interpolation_rotation=mechinterfabric.rotation.average_Manton2004,
        )[0],
        axis=1,
    )

    #############################
    # Plot

    for visualization_method in [
        # mechinterfabric.visualization_matplotlib.plot_projection_of_N4_onto_sphere,
        # mechinterfabric.visualization_matplotlib.plot_approx_FODF_by_N4,
        mechinterfabric.visualization_plotly.add_N4_plotly,
    ]:

        scale = 1.2
        for _, row in new.iterrows():
            visualization_method(
                fig=fig,
                origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
                N4=np.array(row["N4"]),
                # color="yellow",
            )

        for _, row in df.iterrows():
            visualization_method(
                fig=fig,
                origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
                N4=np.array(row["N4"]),
                # color="red",
            )

        # bbox_min = 0
        # bbox_max = 14 * scale
        # ax.auto_scale_xyz(
        #     [bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max]
        # )

        name = (
            str(interpolation_method.__name__)
            + "\n"
            + str(visualization_method.__name__)
        )

        # ax.set_title(name)

        path_picture = os.path.join(directory, name.replace("\n", "_") + ".png")
        fig.write_image(path_picture)

        fig.show()
