import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import mechkit
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem
import os
import matplotlib.pyplot as plt

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s011")
os.makedirs(directory, exist_ok=True)

#########################################################


def to_mandel6(N4s):
    return converter.convert(
        inp=N4s, source="tensor", target="mandel6", quantity="stiffness"
    )


def get_additional_rotation_into_unique_eigensystem(N4_tensor_in_eigen):
    transforms_raw = mechinterfabric.utils.get_orthotropic_sym_rotations(as_dict=True)

    index_d8 = np.s_[0, 0, 0, 1]
    index_d6 = np.s_[0, 0, 0, 2]

    d8 = N4_tensor_in_eigen[index_d8]
    d6 = N4_tensor_in_eigen[index_d6]

    if (0.0 <= d8) and (0.0 <= d6):
        key = "no flip"
    elif (0.0 <= d8) and (d6 < 0.0):
        key = "flip xy"
    elif (d8 < 0.0) and (0.0 <= d6):
        key = "flip xz"
    elif (d8 < 0.0) and (d6 < 0.0):
        key = "flip yz"

    print(key)

    return transforms_raw[key]


def interpolate_N4_decomp_unique_rotation(N4s, weights):
    mechinterfabric.utils.assert_notation_N4(N4s, weights)

    I2 = mechkit.tensors.Basic().I2

    N2s = np.einsum("mijkl,kl->mij", N4s, I2)

    # Get rotations into eigensystem
    eigenvals, rotations_non_unique = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )
    rotations_non_unique = np.array(rotations_non_unique)

    # Rotate each N4 into one possible eigensystem
    N4s_eigen_non_unique = mechinterfabric.interpolation.apply_rotation(
        rotations=rotations_non_unique, tensors=N4s
    )

    # Get unique eigensystem
    additional_rotation = np.array(
        [
            get_additional_rotation_into_unique_eigensystem(N4_tensor_in_eigen=N4_eigen)
            for N4_eigen in N4s_eigen_non_unique
        ]
    )

    rotations = np.einsum(
        "...ij,...jk->...ik", rotations_non_unique, additional_rotation
    )

    # print("non unique:\n", rotations_non_unique)
    # print("additional:\n", additional_rotation)
    # print("resulting:\n", rotations)

    # Rotate each N4 into the unique eigensystem
    N4s_eigen = mechinterfabric.interpolation.apply_rotation(
        rotations=additional_rotation, tensors=N4s_eigen_non_unique
    )

    # # Inspect signs
    # for index in range(len(N4s_eigen)):
    #     print(to_mandel6(N4s_eigen_non_unique[index]))
    #     print(to_mandel6(N4s_eigen[index]))
    #     print()
    #
    # # Inspect combination of rotations
    # N4s_eigen_straight = mechinterfabric.interpolation.apply_rotation(
    #     rotations=rotations, tensors=N4s
    # )
    #
    # assert np.allclose(N4s_eigen, N4s_eigen_straight)
    #
    # for index in range(len(N4s_eigen)):
    #     print('Stepwise vs. straight')
    #     print(to_mandel6(N4s_eigen[index]))
    #     print(to_mandel6(N4s_eigen_straight[index]))
    #     print()

    # Get average rotation
    rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

    # Average components in eigensystems
    N4_av_eigen = np.einsum("m, mijkl->ijkl", weights, N4s_eigen)

    # Rotate back to world COS
    N4_av = mechinterfabric.interpolation.apply_rotation(
        rotations=rotation_av.T, tensors=N4_av_eigen
    )

    # Check if N4_av[I2] == N2_av
    N4_av_I2_eigen = np.einsum("ijkl,kl->ij", N4_av_eigen, I2)
    N2_av_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))
    assert np.allclose(N4_av_I2_eigen, N2_av_eigen)

    return N4_av, N4_av_eigen, rotation_av, N2_av_eigen, N4s_eigen, rotations


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
    ) = interpolate_N4_decomp_unique_rotation(N4s=N4s, weights=weights)

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

    plot_N4 = mechinterfabric.visualization.plot_N4_COS_projection_FODF

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
