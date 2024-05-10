import itertools
import os

import matplotlib.pyplot as plt
import mechkit
import numpy as np
from scipy.spatial.transform import Rotation

import mechinterfabric


np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s010")
os.makedirs(directory, exist_ok=True)

np.random.seed(seed=100)

#########################################################

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


biblio = mechkit.fabric_tensors.Basic().N2

pairs = {
    "special case": (
        np.diag([0.95, 0.05, 0]),
        np.diag([0.0, 0.95, 0.05]),
    ),
    "iso vs ud x": (
        con.to_tensor(biblio["ud_y"]),
        con.to_tensor(biblio["ud_x"]),
    ),
}


def angle_between_two_rotations(rotmatrix_1, rotmatrix_2):

    quat_1 = Rotation.from_matrix(rotmatrix_1).as_quat()
    quat_2 = Rotation.from_matrix(rotmatrix_2).as_quat()

    scalar = np.einsum("i,i->", quat_1, quat_2)

    angle = np.arccos(2.0 * scalar * scalar - 1.0)

    return angle


def get_closest_rotation_matrices(rotmatrix_1, rotmatrix_2):
    variants = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    combinations = list(itertools.combinations(list(range(len(variants))), 2))

    pairs_of_matrices = np.array(
        [
            [
                np.einsum("j, ij->ij", variants[index_1], rotmatrix_1),
                np.einsum("j, ij->ij", variants[index_2], rotmatrix_2),
            ]
            for (index_1, index_2) in combinations
        ]
    )

    angles = np.array(
        [
            angle_between_two_rotations(rotmat_1, rotmat_2)
            for (rotmat_1, rotmat_2) in pairs_of_matrices
        ]
    )

    print(angles)

    index_minimal_angle = angles.argsort()[0]

    closest_1, closest_2 = pairs_of_matrices[index_minimal_angle]

    return closest_1, closest_2


for key, (N2_1, N2_2) in pairs.items():
    print("###########")
    print(key)

    eigenvals, rotation_matrices = zip(
        *[
            mechinterfabric.utils.get_rotation_matrix_into_eigensystem(N2)
            for N2 in [N2_1, N2_2]
        ]
    )

    # angle = angle_between_two_rotations(
    #     rotmatrix_1=rotation_matrices[0], rotmatrix_2=rotation_matrices[1]
    # )

    rotmatrix_1 = rotation_matrices[0]
    rotmatrix_2 = rotation_matrices[1]

    print("Initial")
    print(rotmatrix_1)
    print(rotmatrix_2)

    m_1, m_2 = get_closest_rotation_matrices(rotmatrix_1, rotmatrix_2)

    print("Final")
    print(m_1)
    print(m_2)
