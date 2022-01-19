import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import matplotlib.pyplot as plt
import os

directory = os.path.join("output", "s007")
os.makedirs(directory, exist_ok=True)

for i in range(10):
    nbr = 2
    weights = np.ones((nbr)) / nbr
    rotations = Rotation.random(nbr)
    mean = rotations.mean(weights=weights)

    quats = rotations.as_quat()
    weighted_sum_quat = np.einsum("ij, i", quats, weights)
    weighted_sum = Rotation.from_quat(weighted_sum_quat)

    if not np.allclose(mean.as_quat(), weighted_sum.as_quat()):

        print(f"{i} #############")
        print(f"{'mean as quat':<20}= {mean.as_quat()}")
        print(f"{'weighted sum quats':<20}= {weighted_sum.as_quat()}")

        print(f"{'mean':<20}= {mean.as_rotvec()}")
        print(f"{'weighted sum quats':<20}= {weighted_sum.as_rotvec()}")
        print(f"{'mean as quat':<20}= {mean.as_quat()}")
        print(f"{'weighted sum quats':<20}= {weighted_sum.as_quat()}")

        print(f"{'mean':<20}= {mean.as_rotvec()}")
        print(f"{'weighted sum quats':<20}= {weighted_sum.as_rotvec()}")

        # for elev in [10, 20]:
        #     for azim in [0, 30, 60]:
        elev = 10
        azim = 30

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.view_init(elev=elev, azim=azim)

        mechinterfabric.visualization.plot_bunch_of_cos3D_along_x(
            ax=ax,
            bunch=list(
                map(
                    lambda x: x.as_matrix(),
                    [rotations[0], mean, rotations[1]],
                )
            ),
            shift_y=-0.2,
        )

        mechinterfabric.visualization.plot_bunch_of_cos3D_along_x(
            ax=ax,
            bunch=list(
                map(
                    lambda x: x.as_matrix(),
                    [rotations[0], weighted_sum, rotations[1]],
                )
            ),
            shift_y=0.2,
        )

        title = f"{i:03}" + "_" + f"elev{elev:03}" + "_" + f"azim{azim:03}"
        ax.set_title(title)

        path_picture = os.path.join(
            directory,
            "mean_vs_quaternionWeightedSum" + "_" + title + ".png",
        )
        plt.savefig(path_picture)
