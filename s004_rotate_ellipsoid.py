import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os

directory = os.path.join("output")
os.makedirs(directory, exist_ok=True)


# N1 = np.diag([0.7, 0.2, 0.1])
N1 = np.diag([0.5, 0.3, 0.2])
direction = np.array([1, 0, 0])
rot_vec = 1 / 8 * np.pi * direction / np.linalg.norm(direction)
rotation = Rotation.from_rotvec(rot_vec)
transform = rotation.as_matrix()

N2 = np.einsum("ki, lj, ij->kl", transform, transform, N1)

N1_eigenvals, N1_matrix = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(N1)

N2_eigenvals, N2_matrix = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(N2)

N1_rotation = Rotation.from_matrix(N1_matrix)
N2_rotation = Rotation.from_matrix(N2_matrix)

validation = Rotation.from_matrix(rotation.as_matrix())
assert np.allclose(validation.as_matrix(), rotation.as_matrix())
assert np.allclose(validation.as_rotvec(), rotation.as_rotvec())


print("#######################")
print(transform)
print(N2_rotation.as_matrix())

print(rotation.as_rotvec())
print(N2_rotation.as_rotvec())

func = mechinterfabric.utils.assert_ortho_normal_right_handed
func(N1_rotation.as_matrix())
func(N2_rotation.as_matrix())
func(transform)

# Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Axes labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

###################
ax.cos3D(
    origin=[0, 1, 0],
    length=1 / 2,
    matrix=np.eye(3),
)

origin = [0, 0, 0]
arrow_x = mechinterfabric.visualization.Arrow3D(
    *[0, 0, 0], *rot_vec * 0.3, ec="black", fc="black"
)
ax.add_artist(arrow_x)

ax.cos3D(
    origin=[-0.5, 1, 0],
    length=1 / 2,
    matrix=rotation.as_matrix(),
)

ax.cos3D(
    origin=[0, 1, 0],
    length=1 / 2,
    matrix=N2_matrix,
)
###################


mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[1, 0, 0],
    radii_in_eigen=N1_eigenvals,
    matrix_into_eigen=N1_matrix,
    color="red",
)
mechinterfabric.visualization.plot_ellipsoid(
    ax=ax,
    origin=[1, 0, 0],
    radii_in_eigen=N2_eigenvals,
    matrix_into_eigen=N2_matrix,
    color="green",
)

ax.cos3D(
    origin=[0, 0, 0],
    length=1 / 2,
    matrix=N1_matrix,
)
ax.cos3D(
    origin=[0, 0, 0],
    length=1 / 2,
    matrix=N2_matrix,
)

bbox_min = -1
bbox_max = 2
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

path_picture = os.path.join(directory, "interpolation_N2" + ".png")
plt.savefig(path_picture)


# radii = np.array([0.5, 0.4, 0.1])
# rot_mat = Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])).as_matrix()
# mechinterfabric.visualization.plot_ellipsoid(
#     ax=ax, origin=[2, 2, 2], radii=radii, rotation_matrix=rot_mat, color="blue"
# )
