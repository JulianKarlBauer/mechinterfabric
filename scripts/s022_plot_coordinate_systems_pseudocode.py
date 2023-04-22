import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

import mechinterfabric

###############################################################


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def do_stuff(self):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        self.do_stuff()
        super().draw(renderer)

    def do_3d_projection(self, renderer):
        return self.do_stuff()


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)

###############################################################


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
    ax.scatter(
        *[row["index_x"] * scale, row["index_y"] * scale, 0],
        s=80,
        facecolors="none",
        edgecolors="orange",
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
