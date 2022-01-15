import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.visualization import plot_bunch_of_cos3D_along_x
import os
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

directory = os.path.join("output")
os.makedirs(directory, exist_ok=True)

quat_1 = Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat()
quat_2 = Rotation.from_rotvec(np.pi / 2 * np.array([1, 0, 0])).as_quat()

quat_av = mechinterfabric.rotation.average_quaternion(
    quaternions=np.vstack([quat_1, quat_2]), weights=np.ones(2) / 2
)

###################################
# Plot bunch of rotations

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_bunch_of_cos3D_along_x(
    ax=ax,
    bunch=list(map(lambda x: Rotation.from_quat(x), [quat_1, quat_2, quat_av])),
)

path_picture = os.path.join(directory, "coords" + ".png")
plt.savefig(path_picture)
