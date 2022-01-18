import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.visualization import plot_bunch_of_cos3D_along_x
from mechinterfabric.visualization import plot_stepwise_interpolation_rotations_along_x
import os
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

directory = os.path.join("output")
os.makedirs(directory, exist_ok=True)


quat_pairs = {
    "quarter_x": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])).as_quat(),
    ),
    "quarter_y": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 4 * np.array([0, 1, 0])).as_quat(),
    ),
    "quarter_z": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1])).as_quat(),
    ),
    "half_x": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 2 * np.array([1, 0, 0])).as_quat(),
    ),
    "half_y": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 2 * np.array([0, 1, 0])).as_quat(),
    ),
    "half_z": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 2 * np.array([0, 0, 1])).as_quat(),
    ),
    "quarter_xy": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_rotvec(np.pi / 4 * np.array([1, 1, 0])).as_quat(),
    ),
    "Exchange x and y axes": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_matrix(
            np.array([[0.0, 1.0, -0.0], [1.0, 0.0, -0.0], [0.0, 0.0, -1.0]])
        ).as_quat(),
    ),
    "Exchange x and y axes without signs": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_matrix(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        ).as_quat(),
    ),
}

for key, (quat_1, quat_2) in quat_pairs.items():
    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=np.vstack([quat_1, quat_2]), weights=[2 / 3, 1 / 3]
    )

    ###################################
    # Plot bunch of rotations

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if False:
        # Plot only end points and one intermediate
        plot_bunch_of_cos3D_along_x(
            ax=ax,
            bunch=list(
                map(
                    lambda x: Rotation.from_quat(x).as_matrix(),
                    [quat_1, quat_av, quat_2],
                )
            ),
        )
    elif True:
        # Plot granularly
        plot_stepwise_interpolation_rotations_along_x(
            ax, quat_1, quat_2, nbr_points=7, scale=1, color="green"
        )

    ax.set_title(key)

    path_picture = os.path.join(directory, "coords" + "_" + key + ".png")
    plt.savefig(path_picture)
