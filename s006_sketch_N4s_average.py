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

#########################################################

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


def random_in_between(shape, lower=-1, upper=1):
    return (upper - lower) * np.random.rand(*shape) + lower


from_vectors = mechkit.fabric_tensors.first_kind_discrete

pairs = {
    "Random: many vs many": (
        from_vectors(random_in_between((8, 3))),
        from_vectors(random_in_between((8, 3))),
    ),
    "Random: many vs few": (
        from_vectors(random_in_between((8, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
    "Random: 1 vs 1": (
        from_vectors(random_in_between((1, 3))),
        from_vectors(random_in_between((1, 3))),
    ),
    "Random: 2 vs 2": (
        from_vectors(random_in_between((2, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
    "Random: 3 vs 3": (
        from_vectors(random_in_between((2, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
}

for key, (N4_1, N4_2) in pairs.items():
    print("###########")
    print(key)

    N4s = np.stack([N4_1, N4_2])

    nbr_N4s = len(N4s)
    weights = np.ones((nbr_N4s)) / nbr_N4s

    ########################################################################

    (
        N4_av,
        N4_av_eigen,
        rotation_av,
        N2_av_eigen,
        N4s_eigen,
        rotations,
    ) = mechinterfabric.interpolation.interpolate_N4_decomp(N4s=N4s, weights=weights)

    ##########
    N4_av_mandel = con.to_mandel6(N4_av)
    N4_av_eigen_mandel = con.to_mandel6(N4_av_eigen)

    # N4
    N4s_eigen_mandel = converter.convert(
        inp=N4s_eigen, source="tensor", target="mandel6", quantity="stiffness"
    )

    if False:
        print(N4s_eigen_mandel)
        print(N4_av_eigen_mandel)

        # N2
        print(N2_av_eigen)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plot_N4 = mechinterfabric.visualization.plot_N4

    ################
    plot_N4(
        ax=ax,
        # N4=N4s_eigen_tensor[0],
        N4=N4_1,
        rotation_matrix=rotations[0],  # COS only
        origin=[0, 0, 0],
    )

    ################
    plot_N4(
        ax=ax,
        N4=N4_av,
        rotation_matrix=rotation_av,
        origin=[1, 0, 0],
    )

    ################
    plot_N4(
        ax=ax,
        # N4=N4s_eigen_tensor[1],
        N4=N4_2,
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

    ax.set_title(key)
    path_picture = os.path.join(directory, key + ".png")
    plt.savefig(path_picture)
