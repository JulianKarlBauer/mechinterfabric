import itertools
import os
import pprint

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import mechinterfabric

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s025")
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
    os.path.join("data", "mail_2022_01_31_1124_N2.csv"),
    header=0,
    sep=",",
)
df_N2.columns = df_N2.columns.str.strip()

# Read N4
df_N4 = pd.read_csv(
    os.path.join("data", "mail_2022_01_31_1124_N4.csv"),
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

new["rotation_av"] = new.apply(
    lambda row: mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_extended_return_values(
        N4s=N4s[N4_indices[row.name]],
        weights=bcoords[row.name],
        func_interpolation_rotation=mechinterfabric.rotation.average_Manton2004,
    )[
        2
    ],
    axis=1,
)
df["rotation_av"] = df.apply(
    lambda row: mechinterfabric.utils.get_rotation_matrix_into_unique_N4_eigensystem(
        N4s=np.array([row["N4"]]),
    )[0],
    axis=1,
)


#############################
# Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=90, azim=-90)

# Axes labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

length = 0.7
for _, row in new.iterrows():
    ax.cos3D(
        origin=[row["index_x"], row["index_y"], 0],
        matrix=np.array(row["rotation_av"]),
        length=length,
    )

for _, row in df.iterrows():
    ax.cos3D(
        origin=[row["index_x"], row["index_y"], 0],
        matrix=np.array(row["rotation_av"]),
        length=length,
    )
    ax.scatter(
        *[row["index_x"], row["index_y"], 0],
        s=80,
        facecolors="none",
        edgecolors="orange",
    )

bbox_min = 0
bbox_max = 14
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

name = "coordinate systems"

ax.set_title(name)
fig.tight_layout()

path_picture = os.path.join(directory, name.replace("\n", "_") + ".png")
plt.savefig(path_picture, dpi=300)

# plt.close(fig)
