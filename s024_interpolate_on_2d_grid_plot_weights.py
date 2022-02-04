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

directory = os.path.join("output", "s024")
os.makedirs(directory, exist_ok=True)


converter = mechkit.notation.ExplicitConverter()

#

# Load reference weights
df = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_02_01_1322_weight_data.csv"),
    header=0,
    sep=",",
)
df.columns = df.columns.str.strip()
# df.set_index(["index_x", "index_y"], inplace=True)


#############################
# Plot

fig, axs = plt.subplots(3, 3)

from mpl_toolkits.axes_grid1 import make_axes_locatable

for index_x in range(3):
    for index_y in range(3):
        ax = axs[index_x, index_y]

        indices = [1, 7, 13]
        key_weight = f"w_{indices[index_x]}_{indices[index_y]}"

        weight = df

        # Axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plot = ax.tricontourf(
            df["index_x"], df["index_y"], df[key_weight]
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(plot, cax=cax)

        name = key_weight

        ax.set_title(name)
        fig.tight_layout()


path_picture = os.path.join(directory, "weights" + ".png")
plt.savefig(path_picture, dpi=300)
