import numpy as np
import mechkit
import pandas as pd
import itertools
import os
from scipy.interpolate import interp1d
import mechinterfabric
import matplotlib.pyplot as plt
import pprint

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s023")
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


################
# Load weights

# Load reference weights
df_weights = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_02_01_1322_weight_data.csv"),
    header=0,
    sep=",",
)
df_weights.columns = df_weights.columns.str.strip()

# Sort
df_weights["weights_reference"] = df_weights.apply(
    lambda row: [
        row[f"w_{index_x}_{index_y}"]
        for _, (index_x, index_y) in df[["index_x", "index_y"]].iterrows()
    ],
    axis=1,
)

# Merge
new = new.merge(df_weights)

##############################################


new["rotation_av"] = new.apply(
    lambda row: mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_extended_return_values(
        N4s=N4s,
        # weights=row["weights"],
        weights=row["weights_reference"],
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

scale = 0.4
for _, row in new.iterrows():
    ax.cos3D(
        origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
        matrix=np.array(row["rotation_av"]),
    )

for _, row in df.iterrows():
    ax.cos3D(
        origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
        matrix=np.array(row["rotation_av"]),
    )

bbox_min = 0
bbox_max = 14 * scale
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

name = "coordinate systems"

ax.set_title(name)
fig.tight_layout()

path_picture = os.path.join(directory, name.replace("\n", "_") + ".png")
plt.savefig(path_picture, dpi=300)

# plt.close(fig)
