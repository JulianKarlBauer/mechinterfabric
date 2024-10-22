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

directory = os.path.join("output", "s137")
os.makedirs(directory, exist_ok=True)


converter = mechkit.notation.ExplicitConverter()

#########################################################
datasets = [
    {
        "path_N2": os.path.join("data", "mail_2022_01_31_1124_N2.csv"),
        "path_N4": os.path.join("data", "mail_2022_01_31_1124_N4.csv"),
        "indices": [i + 1 for i in range(13)],
    },
    {
        "path_N2": os.path.join("data", "FOT_field_GF_new_N2.csv"),
        "path_N4": os.path.join("data", "FOT_field_GF_new_N4.csv"),
        "indices": [i + 2 for i in range(11)],
    },
]
dataset = datasets[0]

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
    dataset["path_N2"],
    header=0,
    sep=",",
)
df_N2.columns = df_N2.columns.str.strip()

# Read N4
df_N4 = pd.read_csv(
    dataset["path_N4"],
    header=0,
    sep=",",
)
df_N4.columns = df_N4.columns.str.strip()

# Merge
df = df_N2.merge(df_N4)


df["N2"] = df.apply(N2_from_row, axis=1)
df["N4"] = df.apply(N4_from_row, axis=1)


def analyse(row):
    analysis = mechinterfabric.FOT4Analysis(FOT4=np.array(row["N4"]))
    analysis.analyse()
    return analysis.parameters


names = ["la1", "la2", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"]

df[names] = df.apply(
    analyse,
    axis=1,
    result_type="expand",
)

print(df[["index_x", "index_y"] + names])

N2s = np.array(df["N2"].to_list())
N4s = np.array(df["N4"].to_list())


#########################################################
# Calc entities in df

df_index = df.set_index(["index_x", "index_y"]).index

N4s_df = np.array(df["N4"].to_list())


# New points
indices = dataset["indices"]

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

# # Plot triangulation
# fig = plt.figure()
# plt.triplot(points[:, 0], points[:, 1], tri.simplices)

# plt.plot(points[:, 0], points[:, 1], "o", label="points")
# for index, point in enumerate(points):
#     plt.text(*point, str(index))

# plt.plot(targets[:, 0], targets[:, 1], "x", label="targets")
# for index, point in enumerate(targets):
#     plt.text(*point, str(index))

# plt.legend()
# path_picture = os.path.join(directory, "triangulation" + ".png")
# plt.savefig(path_picture, dpi=300)

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
    # mechinterfabric.interpolation.interpolate_N4_decomp_extended_return_values,
    # mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_extended_return_values,
    mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_analysis_extended_return_values,
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

    class pyplot_3D_annotation_plotter:
        def __init__(self):
            self.annotation_bucket = []

        def plot_tp_ensemble(self, visualization_method, row, scale):

            origin = np.array([row["index_x"] * scale, row["index_y"] * scale, 0])

            visualization_method(
                fig=fig,
                N4=np.array(row["N4"]),
                origin=origin,
                # nbr_points=20,
            )

            position = origin + np.array([0.05, -0.4, 0]) * scale

            text_color = (1, 0, 0)  # Red
            text_color_plotly_rgb = "rgb" + str(
                tuple((np.array(text_color) * 255).tolist())
            )

            self.annotation_bucket.append(
                dict(
                    x=position[0],
                    y=position[1],
                    z=position[2],
                    showarrow=False,
                    text=f"({row['index_x']}, {row['index_y']})",
                    xanchor="left",
                    # xshift=10,
                    # yshift=-10,
                    opacity=0.7,
                    font=dict(
                        color=text_color_plotly_rgb,
                        size=18,
                    ),
                )
            )

            ## Add hollow box around fix points
            import plotly.graph_objects as go
            from itertools import product

            offset = 0.45 * scale

            # structure = np.array(list(product([-1, 1], repeat=3)))
            structure = np.array(
                [
                    [-1, -1, -1],
                    [-1, 1, -1],
                    [1, 1, -1],
                    [1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, 1],
                    [1, 1, 1],
                    [1, -1, 1],
                ]
            )

            corners = origin + structure * offset

            # raise Exception()

            mesh = go.Mesh3d(
                # 8 vertices of a cube
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.05,
                color="black",
                flatshading=True,
            )
            fig.add_trace(mesh)

    plotter = pyplot_3D_annotation_plotter()

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
                # nbr_points=20,
                # color="yellow",
            )

        # for _, row in df.iterrows():
        #     visualization_method(
        #         fig=fig,
        #         origin=[row["index_x"] * scale, row["index_y"] * scale, 0],
        #         N4=np.array(row["N4"]),
        #         # color="red",
        #     )

        for _, row in df.iterrows():
            plotter.plot_tp_ensemble(
                row=row,
                visualization_method=visualization_method,
                scale=scale,
            )

        fig.update_layout(scene=dict(annotations=plotter.annotation_bucket))

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
        # fig.write_image(path_picture)

        fig.show()
