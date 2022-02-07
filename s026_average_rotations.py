import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.visualization import plot_bunch_of_cos3D_along_x
from mechinterfabric.visualization import plot_stepwise_interpolation_rotations_along_x
import os
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s026")
os.makedirs(directory, exist_ok=True)


quat_pairs = {
    "quarter_x": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])),
        Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])),
    ),
}


def interpolate(matrices, weights):
    return matrices[0]


for key, (rot_1, rot_2) in quat_pairs.items():
    print("##########")
    print(key)
    mat_1 = rot_1.as_matrix()
    mat_2 = rot_2.as_matrix()

    frob = np.linalg.norm(mat_1 - mat_2)
    print("frobenius_norm_matrices=", frob)
    print("angle distance = ", 2 * np.arcsin(frob / (2.0 * np.sqrt(2))))

    mat_mean = interpolate(matrices=[mat_1, mat_2], weights=[0.5, 0.5])
    rot_mean = Rotation.from_matrix(mat_mean)

    ###################################
    # Plot bunch of rotations

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot only end points and one intermediate
    plot_bunch_of_cos3D_along_x(
        ax=ax,
        bunch=list(
            map(
                lambda x: x.as_matrix(),
                [rot_1, rot_mean, rot_2],
            )
        ),
    )

    ax.set_title(key)

    path_picture = os.path.join(directory, "coords" + "_" + key + ".png")
    plt.savefig(path_picture)
