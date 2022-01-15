import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.visualization import plot_bunch_of_cos3D_along_x
import os
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

directory = os.path.join("output")
os.makedirs(directory, exist_ok=True)

rot_1 = Rotation.from_rotvec(0 * np.array([1, 0, 0]))
rot_2 = Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0]))

quat_1 = rot_1.as_quat()
quat_2 = rot_2.as_quat()

quat_av = mechinterfabric.rotation.average_quaternion(
    quaternions=np.vstack([quat_1, quat_2]), weights=np.ones(2) / 2
)

rot_av = Rotation.from_quat(quat_av)
mat_av = rot_av.as_matrix()
print(mat_av)


###################################
# Plot bunch of rotations

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_bunch_of_cos3D_along_x(ax=ax, bunch=[rot_1, rot_2, rot_av])

path_picture = os.path.join(directory, "coords" + ".png")
plt.savefig(path_picture)
