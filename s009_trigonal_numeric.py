from pprint import pprint

import mechkit
import numpy as np
import sympy as sp
import vofotensors

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=4)


con = mechkit.notation.Converter()

deviator = mechinterfabric.deviators.deviators[3]
print(deviator.__name__)


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d3, d9

    return sp.lambdify(
        [alpha1, d3, d9],
        vofotensors.fabric_tensors.N4s_parametric["trigonal"]["alpha1_d3_d9"],
    )


fot4 = lambdified_parametrization()(**{"alpha1": 0, "d3": 0.0125, "d9": 0.0325})

# analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
# analysis.get_eigensystem()
# reconstructed = mechinterfabric.utils.rotate_to_mandel("#######################\
#     analysis.FOT4.tensor, analysis.eigensystem
# )
deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))


def inspect(deviator):

    eigenvalues, eigenvectors = np.linalg.eigh(deviator)
    print("\n###########")
    print(eigenvalues)

    for eval, evec in zip(eigenvalues, eigenvectors.T):
        print(f"#######################\nEigenvalue {eval} has eigen tensor")
        tensor = con.to_tensor(evec)
        print(tensor)

        eigenvalues2, eigenvectors2 = np.linalg.eigh(tensor)
        for eval2, evec2 in zip(eigenvalues2, eigenvectors2.T):
            print(f"\t\t Eigenvalue {eval2} has eigen tensor")
            print(f"\t\t{evec2}")

        # raise Exception()
    # print(eigenvectors)


inspect(deviator)
