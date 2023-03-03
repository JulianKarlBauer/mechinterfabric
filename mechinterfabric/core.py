from collections import Counter

import mechkit
import numpy as np
import scipy

from . import decompositions
from . import utils

converter = mechkit.notation.Converter()


class FiberOrientationTensor:
    def __init__(self, FOT):
        self.tensor = converter.to_tensor(FOT)
        self.mandel6 = converter.to_mandel6(FOT)
        self._assert_is_FOT()
        self.deviator = converter.to_mandel6(
            mechkit.operators.dev(self.tensor, order=self.order)
        )

    def _assert_is_FOT(self):
        assert self.tensor.shape == (3,) * self.order

        # Is completely symmetric
        assert np.allclose(mechkit.operators.Sym()(self.tensor), self.tensor)

        # Eigenvalues sum to one
        representation = self.tensor if self.order == 2 else self.mandel6
        assert np.allclose(np.linalg.eigh(representation)[0].sum(), 1.0)


class FiberOrientationTensor2(FiberOrientationTensor):
    def __init__(self, FOT):
        self.order = 2
        super().__init__(FOT)


class FiberOrientationTensor4(FiberOrientationTensor):
    def __init__(self, FOT):
        self.order = 4
        super().__init__(FOT)


class FOT4Analysis:
    def __init__(self, FOT4):
        self.FOT4 = FiberOrientationTensor4(FOT4)
        self.calc_FOT2()

        self._keys_sym_FOT2 = ["isotropic", "transversely_isotropic", "orthotropic"]

    def calc_FOT2(self):
        self.FOT2 = FiberOrientationTensor2(np.tensordot(self.FOT4.tensor, np.eye(3)))
        return self

    def get_symmetry_FOT2(self):
        self.FOT2_spectral_decomposition = decompositions.SpectralDecompositionFOT2(
            FOT2=self.FOT2
        )
        self.FOT2_symmetry = self.FOT2_spectral_decomposition.get_symmetry()

    def get_symmetry_FOT4(self):

        self.FOT4_spectral_decomposition = (
            decompositions.SpectralDecompositionDeviator4(
                FOT4_deviator=self.FOT4.deviator
            )
        )
        self.FOT4_symmetry = self.FOT4_spectral_decomposition.get_symmetry()

    def get_eigensystem(self):

        self.get_symmetry_FOT2()
        self.get_symmetry_FOT4()

        locators = {
            (
                "isotropic_or_cubic",
                "isotropic",
            ): decompositions.EigensystemLocatorIsotropicIsotropic,
            (
                "isotropic_or_cubic",
                "cubic",
            ): decompositions.EigensystemLocatorIsotropicCubic,
            (
                "isotropic_or_cubic",
                "tetragonal",
            ): decompositions.EigensystemLocatorTransvTetraTrigo,
            (
                "isotropic_or_cubic",
                "trigonal or transversely isotropic",
            ): decompositions.EigensystemLocatorTransvTetraTrigo,
        }
        try:
            symmetry_combination = (self.FOT2_symmetry, self.FOT4_symmetry)
            locator = locators[symmetry_combination]
        except KeyError:
            raise utils.ExceptionMechinterfabric(
                f"Locator for symmetry combination {symmetry_combination} not implemented"
            )
        self.eigensystem_locator = locator(
            spectral_decomposition=self.FOT4_spectral_decomposition
        )
        self.eigensystem = self.eigensystem_locator.get_eigensystem()
        return self.eigensystem


# class FourthOrderFabricAnalyser:
#     def analyse(self, FOT4):
#         self.analysis = FOT4Analysis(FOT4)
