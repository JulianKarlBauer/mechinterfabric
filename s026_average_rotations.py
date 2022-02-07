import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.visualization import plot_bunch_of_cos3D_along_x
from mechinterfabric.visualization import plot_stepwise_interpolation_rotations_along_x
import os
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s026")
os.makedirs(directory, exist_ok=True)


quat_pairs = {
    "quarter_x": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])),
        Rotation.from_rotvec(np.pi / 4 * np.array([1, 0, 0])),
    ),
    "Exchange x and y axes without signs": (
        Rotation.from_rotvec(0.001 * np.array([1, 0, 0])),
        Rotation.from_matrix(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        ),
    ),
    "Exchange x and y axes": (
        Rotation.from_rotvec(1e-5 * np.array([1, 0, 0])),
        Rotation.from_matrix(
            np.array([[0.0, 1.0, -0.0], [1.0, 0.0, -0.0], [0.0, 0.0, -1.0]])
        ),
    ),
}


def angle_between_matrices(mat_1, mat_2):
    frob = np.linalg.norm(mat_1 - mat_2)
    angle = 2 * np.arcsin(frob / (2.0 * np.sqrt(2)))
    return angle


for key, (rot_1, rot_2) in quat_pairs.items():
    print("##########")
    print(key)
    mat_1 = rot_1.as_matrix()
    mat_2 = rot_2.as_matrix()

    frob = np.linalg.norm(mat_1 - mat_2)
    angle = angle_between_matrices(mat_1=mat_1, mat_2=mat_2)
    print("frobenius_norm_matrices=", frob)
    print("angle distance = ", angle)

    assert np.isclose(
        angle, mechinterfabric.interpolation.angle_between_two_rotations(mat_1, mat_2)
    )
    ###################################
    # Plot bunch of rotations

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot only end points and one intermediate

    nbr_points = 5
    scale = 1

    single_weights = np.linspace(0, 1, nbr_points)
    weights = np.array([1.0 - single_weights, single_weights]).T
    origins = np.vstack(
        [np.linspace(0, scale, nbr_points), np.zeros((2, nbr_points))]
    ).T

    for index in range(nbr_points):

        mat_mean = mechinterfabric.rotation.average_Manton2004(
            matrices=[mat_1, mat_2], weights=weights[index]
        )
        rot_mean = Rotation.from_matrix(mat_mean)

        ax.cos3D(
            origin=origins[index],
            length=0.3 * scale,
            matrix=mat_mean,
        )

    ax.set_title(key)

    path_picture = os.path.join(directory, "coords" + "_" + key + ".png")
    plt.savefig(path_picture)
