import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os

directory = os.path.join("output", "s002")
os.makedirs(directory, exist_ok=True)


N1 = np.diag([0.95, 0.05, 0])
N2 = np.diag([0.0, 0.95, 0.05])

bunch = np.array([N1, N2])

av, av_in_eigen, av_rotation = mechinterfabric.interpolation.interpolate_N2_decomp(
    bunch, weights=np.ones(len(bunch)) / len(bunch)
)

# Plot

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Axes labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


mechinterfabric.visualization.plot_stepwise_interpolation_along_x(
    ax, N1, N2, nbr_points=5, scale=4
)


bbox_min = -2
bbox_max = 2
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

path_picture = os.path.join(directory, "interpolation_N2" + ".png")
plt.savefig(path_picture)
