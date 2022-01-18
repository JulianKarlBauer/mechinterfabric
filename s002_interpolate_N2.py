import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os

directory = os.path.join("output")
os.makedirs(directory, exist_ok=True)


N1 = np.diag([0.95, 0.05, 0])
N2 = np.diag([0.0, 0.95, 0.05])

bunch = np.array([N1, N2])

av, av_in_eigen, av_rotation = mechinterfabric.interpolation.average_N2(
    bunch, weights=np.ones(len(bunch)) / len(bunch)
)

# Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Axes labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

if False:
    mechinterfabric.visualization.plot_ellipsoid(
        ax=ax,
        origin=[0, 0, 0],
        radii_in_eigen=np.diag(N1),
        matrix_into_eigen=np.eye(3),
        color="red",
    )

    mechinterfabric.visualization.plot_ellipsoid(
        ax=ax,
        origin=[1, 0, 0],
        radii_in_eigen=np.diag(N2),
        matrix_into_eigen=np.eye(3),
        color="green",
    )

    mechinterfabric.visualization.plot_ellipsoid(
        ax=ax,
        origin=[2, 0, 0],
        radii_in_eigen=np.diag(av_in_eigen),
        matrix_into_eigen=av_rotation,
        color="blue",
    )
elif True:
    mechinterfabric.visualization.plot_stepwise_interpolation_along_x(
        ax, N1, N2, nbr_points=5, scale=4, colors=None
    )

bbox_min = -2
bbox_max = 2
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

path_picture = os.path.join(directory, "interpolation_N2" + ".png")
plt.savefig(path_picture)


# radii_in_eigen = np.array([0.5, 0.4, 0.1])
# rot_mat = Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])).as_matrix()
# mechinterfabric.visualization.plot_ellipsoid(
#     ax=ax, origin=[2, 2, 2], radii_in_eigen=radii_in_eigen, matrix_into_eigen=rot_mat, color="blue"
# )
