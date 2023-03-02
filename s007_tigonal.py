from pprint import pprint

import mechkit
import numpy as np
import sympy as sp

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

con = mechkit.notation.ConverterSymbolic()

deviator = mechinterfabric.deviators.deviators[3]
print(deviator.__name__)


def inspect(deviator):
    matrix = sp.Matrix(deviator)

    eigenvalues = matrix.eigenvals()
    print("\n###########")
    print(eigenvalues)

    eigenvectors = matrix.eigenvects()
    for ev in eigenvectors:
        print(f"{ev[0]} with multiplicity={ev[1]}")
        for eigenspace in ev[2]:
            nicer = np.array([row for row in eigenspace])
            print(con.to_tensor(nicer))

            eigenvectors_eigenspace = sp.Matrix(
                con.to_tensor(np.array(eigenspace.tolist())[:, 0])
            ).eigenvects()
            for ev_eigenspace in eigenvectors_eigenspace:
                print(f"\t{ev_eigenspace[0]} with multiplicity={ev_eigenspace[1]}")
                for eigenspace_eigenspace in ev_eigenspace[2]:
                    nicer_eigenspace = np.array([row for row in eigenspace_eigenspace])
                    print(f"\t{nicer_eigenspace}")

            # raise Exception()
    # print(eigenvectors)


inspect(deviator())
