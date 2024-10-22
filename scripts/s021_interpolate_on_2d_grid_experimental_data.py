import itertools
import os
import pprint

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import mechinterfabric.visualization_matplotlib

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s021")
os.makedirs(directory, exist_ok=True)

directory_data = os.path.join("export")
os.makedirs(directory_data, exist_ok=True)

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

N4s_mandel = converter.convert(
    inp=N4s, source="tensor", target="mandel6", quantity="stiffness"
)

I2 = mechkit.tensors.Basic().I2
for index, N2 in enumerate(N2s):
    N2_from_N4 = np.einsum("ijkl,kl->ij", N4s[index], I2)
    assert np.allclose(N2, N2_from_N4, atol=1e-5)

#########################################################

dimensions = {
    "x": {1: 0.0, 7: 196.0, 13: 379.0},
    "y": {1: 0.0, 7: 197.0, 13: 379.0},
}
index_x_to_x = interp1d(
    list(dimensions["x"].keys()), list(dimensions["x"].values()), kind="linear"
)
index_y_to_y = interp1d(
    list(dimensions["y"].keys()), list(dimensions["y"].values()), kind="linear"
)


def distance_shepard_inverse(row_1, row_2):
    difference = np.array([row_1["x"] - row_2["x"], row_1["y"] - row_2["y"]])
    return 1.0 / np.sqrt((difference * difference).sum())


#########################################################

# Calc entities in df
df_index = df.set_index(["index_x", "index_y"]).index
# df.set_index(["index_x", "index_y"], inplace=True)

df["x"] = df.apply(lambda row: index_x_to_x(row["index_x"]), axis=1)
df["y"] = df.apply(lambda row: index_y_to_y(row["index_y"]), axis=1)

N4s_df = np.array(df["N4"].to_list())


# New points
indices = [i + 1 for i in range(13)]
# indices = [i + 1 for i in range(13) if i in [0, 3, 6, 9, 12]]

indices_points = list(itertools.product(indices, repeat=2))
indices_points = [index for index in indices_points if index not in df_index]

new = pd.DataFrame(indices_points, columns=["index_x", "index_y"])


new["x"] = new.apply(lambda row: index_x_to_x(row["index_x"]), axis=1)
new["y"] = new.apply(lambda row: index_y_to_y(row["index_y"]), axis=1)


def get_normalized_weigths(row, df_reference_points):
    weights = [
        distance_shepard_inverse(row_1=row, row_2=reference_row)
        for _, reference_row in df_reference_points.iterrows()
    ]
    sum_weights = np.sum(weights)
    return weights / sum_weights


new["weights"] = new.apply(
    lambda row: get_normalized_weigths(row=row, df_reference_points=df),
    axis=1,
)


def check_if_all_weight_sets_are_normalized(weights):
    return np.allclose(np.array(weights.to_list()).sum(axis=1), np.ones((len(weights))))


# Check whether all weight sets are normalized
assert check_if_all_weight_sets_are_normalized(weights=new["weights"])

################
# Compare weights

# Load reference weights
df_weights = pd.read_csv(
    os.path.join("data", "mail_2022_02_01_1322_weight_data.csv"),
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

# Check whether all weight sets are normalized
assert check_if_all_weight_sets_are_normalized(weights=df_weights["weights_reference"])

# Merge
new = new.merge(df_weights)

#####
# Export

df[["index_x", "index_y", "x", "y"]].to_csv(os.path.join(directory_data, "df.csv"))
new[["index_x", "index_y", "x", "y", "weights", "weights_reference"]].to_csv(
    os.path.join(directory_data, "new.csv")
)


def multidim_array_to_file(array, path, array_name="data"):
    with np.printoptions(threshold=np.inf):
        with open(path, "wt") as f:
            f.write("from numpy import array\n")
            f.write(f"{array_name}=")
            pprint.pprint(array, stream=f)


multidim_array_to_file(
    array=np.array(df["N4"].to_list()),
    path=os.path.join(directory_data, "df_N4s" + ".py"),
)


##############################################

for interpolation_method in [
    mechinterfabric.interpolation.interpolate_N4_decomp,
    mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation,
]:

    new["N4"] = new.apply(
        lambda row: interpolation_method(
            N4s=N4s,
            # weights=row["weights"],
            weights=row["weights_reference"],
        ).tolist(),
        axis=1,
    )

    # Export
    multidim_array_to_file(
        array=np.array(new["N4"].to_list()),
        path=os.path.join(
            directory_data, "new_N4s" + "_" + str(interpolation_method.__name__) + ".py"
        ),
    )

    ########################
    # Validate case with single non-vanishing weight

    def get_singgular_weights(index):
        zeros = [
            0.0,
        ] * len(df)
        zeros[index] = 1.0
        return zeros

    validation = [
        interpolation_method(
            N4s=N4s, weights=get_singgular_weights(index=index)
        ).tolist()
        for index, (_, row) in enumerate(df.iterrows())
    ]
    assert np.allclose(N4s, np.array(validation))

    N4s_mandel_new = converter.convert(
        inp=np.array(new["N4"].to_list()),
        source="tensor",
        target="mandel6",
        quantity="stiffness",
    )
    #############################
    # Plot

    for visualization_method in [
        # mechinterfabric.visualization_matplotlib.plot_projection_of_N4_onto_sphere,
        mechinterfabric.visualization_matplotlib.plot_approx_FODF_by_N4,
    ]:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=90, azim=-90)

        # Axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        scale = 1.2
        for _, row in new.iterrows():
            visualization_method(
                ax=ax,
                origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
                N4=np.array(row["N4"]),
                color="yellow",
            )

        for _, row in df.iterrows():
            visualization_method(
                ax=ax,
                origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
                N4=np.array(row["N4"]),
                color="red",
            )

        bbox_min = 0
        bbox_max = 14 * scale
        ax.auto_scale_xyz(
            [bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max]
        )

        name = (
            str(interpolation_method.__name__)
            + "\n"
            + str(visualization_method.__name__)
        )

        ax.set_title(name)
        fig.tight_layout()

        path_picture = os.path.join(directory, name.replace("\n", "_") + ".png")
        plt.savefig(path_picture, dpi=300)

    # plt.close(fig)
