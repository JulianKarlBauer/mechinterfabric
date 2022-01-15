import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


N1 = np.diag([0.95, 0.05, 0])
N2 = np.diag([0.3, 0.3, 0.3])

N1_eigenvals, N1_rotations = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(
    N1
)

N2_eigenvals, N2_rotations = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(
    N2
)

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


mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[0, 0, 0],
    radii=N1_eigenvals,
    rotation_matrix=N2_rotations,
    color="red",
)

mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[1, 0, 0],
    radii=N2_eigenvals,
    rotation_matrix=N2_rotations,
    color="green",
)

mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[2, 0, 0],
    radii=np.diag(av_in_eigen),
    rotation_matrix=av_rotation,
    color="blue",
)


# radii = np.array([0.5, 0.4, 0.1])
# rot_mat = Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])).as_matrix()
# mechinterfabric.visualization.plot_ellipsoid(
#     ax=ax, origin=[2, 2, 2], radii=radii, rotation_matrix=rot_mat, color="blue"
# )
