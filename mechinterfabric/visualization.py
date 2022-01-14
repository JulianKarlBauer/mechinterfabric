from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)

#################################


def _cos3D(ax, origin, length, rotation, *args, **kwargs):
    # origin = np.array([x, y, z])

    matrix = rotation.as_matrix()

    arrow_x = np.array([1.0, 0, 0]) * length
    arrow_y = np.array([0, 1.0, 0]) * length
    arrow_z = np.array([0.0, 0, 1.0]) * length

    dx_x, dy_x, dz_x = np.einsum("ij, j->i", matrix, arrow_x)
    dx_y, dy_y, dz_y = np.einsum("ij, j->i", matrix, arrow_y)
    dx_z, dy_z, dz_z = np.einsum("ij, j->i", matrix, arrow_z)

    arrow_x = Arrow3D(
        origin[0],
        origin[1],
        origin[2],
        dx_x,
        dy_x,
        dz_x,
        *args,
        ec="red",
        fc="red",
        **kwargs
    )
    arrow_y = Arrow3D(
        origin[0],
        origin[1],
        origin[2],
        dx_y,
        dy_y,
        dz_y,
        *args,
        ec="green",
        fc="green",
        **kwargs
    )
    arrow_z = Arrow3D(
        origin[0],
        origin[1],
        origin[2],
        dx_z,
        dy_z,
        dz_z,
        *args,
        ec="blue",
        fc="blue",
        **kwargs
    )

    ax.add_artist(arrow_x)
    ax.add_artist(arrow_y)
    ax.add_artist(arrow_z)


setattr(Axes3D, "cos3D", _cos3D)

########################
# Application


def plot_bunch_along_x(ax, bunch):

    origins = np.linspace(0, 1, len(bunch))

    factor = 5.0
    length = factor / (len(bunch) - 1) / (1.0 + 2 * factor)

    offset = length / factor
    ax.set_xlim(0 - offset, 1 + offset)
    ax.set_ylim(-0.5 - offset, 0.5 + offset)
    ax.set_zlim(-0.5 - offset, 0.5 + offset)

    for index, rot in enumerate(bunch):
        ax.cos3D(
            origin=[origins[index], 0, 0],
            length=length,
            rotation=rot,
        )

    return ax
