import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


N1 = np.diag([0.95, 0.05, 0])
N2 = np.diag([0, 0.95, 0.05])

N1_eigenvals, N1_rotations = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(
    N1
)

N2_eigenvals, N2_rotations = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(
    N2
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