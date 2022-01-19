import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import os
import matplotlib.pyplot as plt
import mechkit
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem
import os
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s006")
os.makedirs(directory, exist_ok=True)

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


def random_in_between(shape, lower=-1, upper=1):
    return (upper - lower) * np.random.rand(*shape) + lower


N4_1_tensor = mechkit.fabric_tensors.first_kind_discrete(random_in_between((8, 3)))
N4_2_tensor = mechkit.fabric_tensors.first_kind_discrete(random_in_between((3, 3)))

N4s_tensor = np.stack([N4_1_tensor, N4_2_tensor])

nbr_N4s = len(N4s_tensor)
weights = np.ones((nbr_N4s)) / nbr_N4s

N4s = converter.convert(
    inp=N4s_tensor, source="tensor", target="mandel6", quantity="stiffness"
)

assert N4s.shape == (len(weights), 6, 6)

# mechkit.notation.converter().to_mandel6(mechkit.tensors.Basic().I2)
I2_mandel6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

N2s_mandel = np.einsum("mij,j->mi", N4s, I2_mandel6)

N2s = converter.convert(
    inp=N2s_mandel, source="mandel6", target="tensor", quantity="stress"
)

############
# Get representations in eigensystems

eigenvals, rotations = zip(*[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s])
rotations = np.array(rotations)

# Average with scipy.spatila.transform.Rotation().mean()
rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

N4s_eigen_tensor = np.einsum(
    "...mi, ...nj, ...ok, ...pl, ...mnop->...ijkl",
    rotations,
    rotations,
    rotations,
    rotations,
    N4s_tensor,
)

N4s_eigen = converter.convert(
    inp=N4s_eigen_tensor, source="tensor", target="mandel6", quantity="stiffness"
)

N4_av_eigen = np.einsum("i, ikl->kl", weights, N4s_eigen)

N2_from_N4_av_eigen = con.to_tensor(np.einsum("ij,j->i", N4_av_eigen, I2_mandel6))
N2_av_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))

print(N4s_eigen)
print(N4_av_eigen)

print(N2_from_N4_av_eigen)
print(N2_av_eigen)

assert np.allclose(N2_from_N4_av_eigen, N2_av_eigen)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

offset_coord = np.array([0, 1, 0])

origin = [0, 0, 0]
mechinterfabric.visualization.plot_projection_of_N4_onto_sphere(
    ax, origin=origin, N4=N4_1_tensor
)
ax.cos3D(origin=origin + offset_coord, matrix=rotations[0])

origin = [1, 0, 0]
mechinterfabric.visualization.plot_projection_of_N4_onto_sphere(
    ax, origin=origin, N4=N4_av_eigen
)
ax.cos3D(origin=origin + offset_coord, matrix=rotation_av)

origin = [2, 0, 0]
mechinterfabric.visualization.plot_projection_of_N4_onto_sphere(
    ax, origin=origin, N4=N4_2_tensor
)
ax.cos3D(origin=origin + offset_coord, matrix=rotations[1])


upper = 2
lower = 0
offset = 0.5
limits = [
    (lower - offset, upper + offset),
    (-0.5 - offset, 0.5 + offset),
    (-0.5 - offset, 0.5 + offset),
]
ax.set_xlim(limits[0])
ax.set_ylim(limits[1])
ax.set_zlim(limits[2])

# Homogeneous axes
bbox_min = np.min(limits)
bbox_max = np.max(limits)
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

path_picture = os.path.join(directory, "main" + ".png")
plt.savefig(path_picture)
