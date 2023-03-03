from pprint import pprint

import mechkit
import numpy as np
import scipy.spatial
import sympy as sp
import vofotensors
from vofotensors.abc import alpha1
from vofotensors.abc import rho1

import mechinterfabric

factor_alpha1, factor_rho1 = sp.symbols("factor_alpha1 factor_rho1", real=True)

np.set_printoptions(linewidth=100000)

converter = mechkit.notation.Converter()

parametrization_symbolic = vofotensors.fabric_tensors.N4s_parametric[
    "transv_isotropic"
]["alpha1_rho1"]
parametrization = sp.lambdify([alpha1, rho1], parametrization_symbolic)

################################################
def rotate(mandel, Q):
    return converter.to_mandel6(
        mechinterfabric.utils.rotate(converter.to_tensor(mandel), Q=Q)
    )


################################################

FOT4 = parametrization(0, 1 / 60)

######
# Rotate

# angle = 52
# rotation_vector = np.array([0, 1, 2])
# rotation = scipy.spatial.transform.Rotation.from_rotvec(
#     angle * rotation_vector, degrees=True
# )
# Q = rotation.as_matrix()
# FOT4_rotated = mechinterfabric.utils.rotate_mandel(FOT4, Q=Q)

FOT4_rotated = mechinterfabric.utils.rotate_fot4_randomly(FOT4)

print(FOT4)
print(FOT4_rotated)


analysis = mechinterfabric.FOT4Analysis(FOT4_rotated)
analysis.get_eigensystem()
decomposition = analysis.FOT4_spectral_decomposition

reconstructed = mechinterfabric.utils.rotate_to_mandel(
    analysis.FOT4.tensor, analysis.eigensystem
)
print(reconstructed)

assert np.allclose(reconstructed, FOT4)
