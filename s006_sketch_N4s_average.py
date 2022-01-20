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


########################################################################

(
    N4_av,
    N4_av_eigen,
    rotation_av,
    N2_av_eigen,
    N4s_eigen,
    rotations,
) = mechinterfabric.interpolation.average_N4(N4s=N4s_tensor, weights=weights)


##########
N4_av_mandel = con.to_mandel6(N4_av)
N4_av_eigen_mandel = con.to_mandel6(N4_av_eigen)

# N4
N4s_eigen_mandel = converter.convert(
    inp=N4s_eigen, source="tensor", target="mandel6", quantity="stiffness"
)
print(N4s_eigen_mandel)
print(N4_av_eigen_mandel)

# N2
print(N2_av_eigen)


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def plot_N4(
    N4,
    rotation_matrix,
    origin=[0, 0, 0],
    offset_coord=[0, 0.5, 0],
    offset_alternative=[0, -0.5, 0],
):
    mechinterfabric.visualization.plot_projection_of_N4_onto_sphere(
        ax, origin=origin, N4=N4
    )
    mechinterfabric.visualization.plot_approx_FODF_by_N4(
        ax, origin=origin + np.array(offset_alternative), N4=N4
    )
    ax.cos3D(origin=origin + np.array(offset_coord), matrix=rotation_matrix)


################
plot_N4(
    # N4=N4s_eigen_tensor[0],
    N4=N4_1_tensor,
    rotation_matrix=rotations[0],
    origin=[0, 0, 0],
)

################
plot_N4(
    N4=N4_av,
    rotation_matrix=rotation_av,
    origin=[1, 0, 0],
)

################
plot_N4(
    # N4=N4s_eigen_tensor[1],
    N4=N4_2_tensor,
    rotation_matrix=rotations[1],
    origin=[2, 0, 0],
)
################


upper = 2
lower = 0
offset = 1
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
