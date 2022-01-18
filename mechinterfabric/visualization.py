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


def _cos3D(ax, origin, length, matrix, *args, **kwargs):
    # origin = np.array([x, y, z])

    arrow_x = np.array([1.0, 0, 0]) * length
    arrow_y = np.array([0, 1.0, 0]) * length
    arrow_z = np.array([0.0, 0, 1.0]) * length

    diff_x = np.einsum("ij, j->i", matrix, arrow_x)
    diff_y = np.einsum("ij, j->i", matrix, arrow_y)
    diff_z = np.einsum("ij, j->i", matrix, arrow_z)

    arrow_x = Arrow3D(
        origin[0], origin[1], origin[2], *diff_x, *args, ec="red", fc="red", **kwargs
    )
    arrow_y = Arrow3D(
        origin[0],
        origin[1],
        origin[2],
        *diff_y,
        *args,
        ec="green",
        fc="green",
        **kwargs
    )
    arrow_z = Arrow3D(
        origin[0], origin[1], origin[2], *diff_z, *args, ec="blue", fc="blue", **kwargs
    )

    ax.add_artist(arrow_x)
    ax.add_artist(arrow_y)
    ax.add_artist(arrow_z)


setattr(Axes3D, "cos3D", _cos3D)

########################
# Application


def plot_bunch_of_cos3D_along_x(ax, bunch):

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
            matrix=rot,
        )

    return ax


def plot_ellipsoid(
    ax,
    origin,
    radii_in_eigen,
    matrix_into_eigen,
    *args,
    nbr_points=40,
    homogeneous_axes=False,
    **kwargs
):

    phi = np.linspace(0.0, 2.0 * np.pi, nbr_points)
    theta = np.linspace(0.0, np.pi, nbr_points)
    x = radii_in_eigen[0] * np.outer(np.cos(phi), np.sin(theta))
    y = radii_in_eigen[1] * np.outer(np.sin(phi), np.sin(theta))
    z = radii_in_eigen[2] * np.outer(np.ones_like(phi), np.cos(theta))

    vectors = np.array([x, y, z])

    # Transform
    vectors = (
        np.einsum("ij, j...->i...", matrix_into_eigen, vectors)
        + np.array(origin)[:, np.newaxis, np.newaxis]
    )

    ax.plot_surface(
        *vectors, rstride=3, cstride=3, linewidth=0.1, alpha=0.5, shade=True, **kwargs
    )

    if homogeneous_axes:
        # Homogeneous axes
        bbox_min = np.min([x, y, z])
        bbox_max = np.max([x, y, z])
        ax.auto_scale_xyz(
            [bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max]
        )
