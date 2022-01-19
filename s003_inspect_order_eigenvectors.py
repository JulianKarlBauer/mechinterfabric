import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os

directory = os.path.join("output", "s003")
os.makedirs(directory, exist_ok=True)


N1 = np.diag([0.95, 0.05, 0])
N2 = np.diag([0, 0.95, 0.05])

N1_eigenvals, N1_matrix = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(N1)
N2_eigenvals, N2_matrix = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(N2)
print(f"N2_eigenvals=\n{N2_eigenvals}\n N2_matrix=\n{N2_matrix}")

################################################

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
    radii_in_eigen=N1_eigenvals,
    matrix_into_eigen=N1_matrix,
    color="red",
)

ax.cos3D(origin=[0, 0, 0], length=1 / 2, matrix=N1_matrix)

mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[1, 0, 0],
    radii_in_eigen=N2_eigenvals,
    matrix_into_eigen=N2_matrix,
    color="green",
)

ax.cos3D(origin=[1, 0, 0], length=1 / 2, matrix=N2_matrix)

#########
# svd

u1, s1, vh1 = np.linalg.svd(N1)
u2, s2, vh2 = np.linalg.svd(N2)
print(f"u2=\n{u2}\n s2=\n{s2}\n vh2=\n{vh2}")

mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[0, 1, 0],
    radii_in_eigen=s1,
    matrix_into_eigen=u1,
    color="blue",
)

mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[1, 1, 0],
    radii_in_eigen=s2,
    matrix_into_eigen=u2,
    color="orange",
)

bbox_min = -2
bbox_max = 2
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

path_picture = os.path.join(directory, "main" + ".png")
plt.savefig(path_picture)
