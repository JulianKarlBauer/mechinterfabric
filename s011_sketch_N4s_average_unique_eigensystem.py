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


def interpolate_N4_decomp_unique_rotation(N4s, weights, closest_eigensystems=False):
    mechinterfabric.utils.assert_notation_N4(N4s, weights)

    I2 = mechkit.tensors.Basic().I2

    N2s = np.einsum("mijkl,kl->mij", N4s, I2)

    # Get rotations into eigensystem
    eigenvals, rotations = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )
    rotations = np.array(rotations)

    if closest_eigensystems:
        rotations = np.array(
            mechinterfabric.interpolation.get_closest_rotation_matrices(*rotations)
        )

    # Get average rotation
    rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

    # Rotate each N4 into it's eigensystem
    N4s_eigen = mechinterfabric.interpolation.apply_rotation(
        rotations=rotations, tensors=N4s
    )

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
