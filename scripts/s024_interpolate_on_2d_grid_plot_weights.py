import itertools
import os
import pprint

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d

import mechinterfabric


np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s024")
os.makedirs(directory, exist_ok=True)


converter = mechkit.notation.ExplicitConverter()

#

# Load reference weights
df = pd.read_csv(
    os.path.join("data", "jb_mail_2022_02_01_1322_weight_data.csv"),
    header=0,
    sep=",",
)
df.columns = df.columns.str.strip()
# df.set_index(["index_x", "index_y"], inplace=True)


#############################
# Plot

fig, axs = plt.subplots(3, 3)

weights = df[
    [
        "w_1_1",
        "w_1_7",
        "w_1_13",
        "w_7_1",
        "w_7_7",
        "w_7_13",
        "w_13_1",
        "w_13_7",
        "w_13_13",
    ]
].to_numpy()
mask = weights.max(axis=1) <= 0.33
masked = df[mask]

for index_x in range(3):
    for index_y in range(3):
        ax = axs[index_x, index_y]

        indices = [1, 7, 13]
        key_weight = f"w_{indices[index_x]}_{indices[index_y]}"

        # Axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plot = ax.tricontourf(df["index_x"], df["index_y"], df[key_weight])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.scatter(masked["index_x"], masked["index_y"], s=6)

        plt.colorbar(plot, cax=cax)

        name = key_weight

        ax.set_title(name)
        fig.tight_layout()


path_picture = os.path.join(directory, "weights" + ".png")
plt.savefig(path_picture, dpi=300)
