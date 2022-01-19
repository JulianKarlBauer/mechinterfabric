import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import os
import matplotlib.pyplot as plt
import mechkit
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem

np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s006")
os.makedirs(directory, exist_ok=True)

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()

N4_1_tensor = mechkit.fabric_tensors.first_kind_discrete(np.random.rand(1, 3))
N4_2_tensor = mechkit.fabric_tensors.first_kind_discrete(np.random.rand(1, 3))

N4s_tensor = np.stack([N4_1_tensor, N4_2_tensor])

nbr_N4s = len(N4s_tensor)
weights = np.ones((nbr_N4s)) / nbr_N4s

N4s = converter.convert(
    inp=N4s_tensor, source="tensor", target="mandel6", quantity="stiffness"
)

assert N4s.shape == (len(weights), 6, 6)

# mechkit.notation.converter().to_mandel6(mechkit.tensors.Basic().I2)
I2_mandel6 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

N2s_mandel = np.einsum("mij,j->mi", N4s, I2_mandel6)

N2s = converter.convert(
    inp=N2s_mandel, source="mandel6", target="tensor", quantity="stress"
)

############
# Get representations in eigensystems

eigenvals, rotations = zip(*[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s])
rotations = np.array(rotations)

# Average with scipy.spatila.transform.Rotation().mean()
rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

N4s_eigen_tensor = np.einsum(
    "...mi, ...nj, ...ok, ...pl, ...mnop->...ijkl",
    rotations,
    rotations,
    rotations,
    rotations,
    N4s_tensor,
)

N4s_eigen = converter.convert(
    inp=N4s_tensor, source="tensor", target="mandel6", quantity="stiffness"
)
print(N4s_eigen)

N4_av_eigen = np.einsum("i, ikl->kl", weights, N4s_eigen)

N2_from_N4_av_eigen = con.to_tensor(np.einsum("ij,j->i", N4_av_eigen, I2_mandel6))
N2_av_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))

print(N2_from_N4_av_eigen)
print(N2_av_eigen)

# return N2_av, N2_av_in_eigen, rotation_av
