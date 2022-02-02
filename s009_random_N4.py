import numpy as np
import mechinterfabric
import os
import matplotlib.pyplot as plt
import mechkit


np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s009")
os.makedirs(directory, exist_ok=True)

np.random.seed(seed=100)

#########################################################

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


def random_in_between(shape, lower=-1, upper=1):
    return (upper - lower) * np.random.rand(*shape) + lower


def from_vectors(vectors):
    return mechkit.fabric_tensors.first_kind_discrete(vectors)


biblio = mechkit.fabric_tensors.Basic().N4

pairs = {
    f"Random: {length}": from_vectors(random_in_between((length, 3)))
    for length in [1, 2, 3, 4, 5, 6]
}

for key, N4 in pairs.items():

    I2 = mechkit.tensors.Basic().I2
    N2 = np.einsum("ijkl,kl->ij", N4, I2)
    (
        N2_eigen_diag,
        rotation_into_eigen,
    ) = mechinterfabric.utils.get_rotation_matrix_into_eigensystem(N2)

    N4_eigen = mechinterfabric.utils.apply_rotation(
        rotations=rotation_into_eigen, tensors=N4
    )

    slice = np.s_[0:3, 3:6]
    # slice = np.s_[:, :]
    selected_positions = np.s_[[0, 0], [4, 5]]

    print("###########")
    print(key)
    print(con.to_mandel6(N4))
    print("Eigen")
    print(con.to_mandel6(N4_eigen))

    rotations = mechinterfabric.utils.get_orthotropic_sym_rotations(as_dict=True)
    # rotations.pop("{v_i}_1: no flip")

    for label, rot in rotations.items():
        N4_eigen_transformed = mechinterfabric.utils.apply_rotation(
            rotations=rot, tensors=N4_eigen
        )
        N4_eigen_transformed_mandel = con.to_mandel6(N4_eigen_transformed)
        if np.allclose(N4_eigen_transformed_mandel[selected_positions], np.zeros(2)):
            print("not valid")
            break
        else:
            if np.all(N4_eigen_transformed_mandel[selected_positions] > 0):
                print(label)
                print(N4_eigen_transformed_mandel[slice])
