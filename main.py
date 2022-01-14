import numpy as np
from scipy.spatial.transform import Rotation
from mechinterfabric.visualization import plot_bunch_along_x
import os
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

directory = os.path.join("output")
os.makedirs(directory, exist_ok=True)


def xyzw_to_wxyz(inp):
    # return np.einsum("ijkl->lijk", inp)
    inp[[0, 1, 2, 3]] = inp[[3, 0, 1, 2]]
    return inp


assert np.allclose(xyzw_to_wxyz(np.array([1, 2, 3, 4])), np.array([4, 1, 2, 3]))

rot_1 = Rotation.from_rotvec(0 * np.array([1, 0, 0]))
rot_2 = Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0]))

print(rot_1.as_matrix())
print(rot_2.as_matrix())

quat_1 = rot_1.as_quat()
quat_2 = rot_2.as_quat()

bunch = [quat_1, quat_2]
# # Adjust notation
# bunch = list(map(xyzw_to_wxyz, bunch))
bunch = np.vstack(bunch)

###################################
# Plot bunch of rotations

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_bunch_along_x(ax=ax, bunch=[rot_1, rot_2, rot_1, rot_2])

path_picture = os.path.join(directory, "coords" + ".png")
plt.savefig(path_picture)
