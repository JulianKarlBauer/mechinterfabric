import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.visualization import plot_bunch_of_cos3D_along_x
from mechinterfabric.visualization import plot_stepwise_interpolation_rotations_along_x
import os
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=100000)

directory = os.path.join("output", 's005')
os.makedirs(directory, exist_ok=True)


quat_pairs = {
    "Exchange x and y axes without signs": (
        Rotation.from_rotvec(0 * np.array([1, 0, 0])).as_quat(),
        Rotation.from_matrix(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        ).as_quat(),
    ),
}

for key, (quat_1, quat_2) in quat_pairs.items():
    print("##########")
    print(key)
    print(quat_1, quat_2)
    mat_1 = Rotation.from_quat(quat_1).as_matrix()
    mat_2 = Rotation.from_quat(quat_2).as_matrix()

    print(mat_1)
    print(mat_2)
    print("frobenius_norm_matrices=", np.linalg.norm(mat_1 - mat_2))

    print()

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
            ax,
            quat_1,
            quat_2,
            nbr_points=5,
            scale=1,
            color="green",
            verbose=True,
            homebreu=True,
        )

    ax.set_title(key)

    path_picture = os.path.join(directory, "coords" + "_" + key + ".png")
    plt.savefig(path_picture)
