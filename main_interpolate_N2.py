import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


N1 = np.diag([1, 0, 0])
N2 = np.diag([0.3, 0.3, 0.3])

bunch = np.array([N1, N2])

av, av_in_eigen, av_rotation = mechinterfabric.interpolation.average_N2(
    bunch, weights=np.ones(len(bunch)) / len(bunch)
)

# Plot


def plot_ellipsoid(
    ax, radii, rotation_matrix, *args, nbr_points=40, homogeneous_axes=True, **kwargs
):

    phi = np.linspace(0.0, 2.0 * np.pi, nbr_points)
    theta = np.linspace(0.0, np.pi, nbr_points)
    x = radii[0] * np.outer(np.cos(phi), np.sin(theta))
    y = radii[1] * np.outer(np.sin(phi), np.sin(theta))
    z = radii[2] * np.outer(np.ones_like(phi), np.cos(theta))

    vectors = np.array([x, y, z])

    # Transform
    vectors = np.einsum("ij, j...->i...", rot_mat, vectors)

    ax.plot_surface(
        *vectors, rstride=3, cstride=3, linewidth=0.1, alpha=1, shade=True, **kwargs
    )

    if homogeneous_axes:
        # Homogeneous axes
        bbox_min = np.min([x, y, z])
        bbox_max = np.max([x, y, z])
        ax.auto_scale_xyz(
            [bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max]
        )


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Axes labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


radii = np.array([0.5, 0.4, 0.1])
rot_mat = Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])).as_matrix()

plot_ellipsoid(ax=ax, radii=radii, rotation_matrix=rot_mat, color="red")
