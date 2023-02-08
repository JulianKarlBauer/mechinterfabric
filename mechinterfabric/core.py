import numpy as np
from collections import Counter
import mechkit
from . import utils
from . import decompositions
import scipy

converter = mechkit.notation.Converter()


class FOT4Analysis:
    def __init__(self, FOT4):
        self._assert_is_FOT4(FOT4)
        self.FOT4_mandel6 = converter.to_mandel6(FOT4)
        self.FOT4_tensor = converter.to_tensor(FOT4)

        self._keys_sym_FOT2 = ["isotropic", "transversely_isotropic", "orthotropic"]

    def calc_FOT2(self):
        I2 = np.eye(3)
        self.FOT2 = np.tensordot(self.FOT4_tensor, I2)
        return self

    def get_symmetry_FOT2(self):
        self.FOT2_spectral_decomposition = decompositions.SpectralDecompositionFOT2(
            FOT2=self.FOT2
        )
        self.FOT2_symmetry = self.FOT2_spectral_decomposition.get_symmetry()

    def calc_FOT4_deviator(self):
        self.FOT4_mandel6_dev = self._get_deviator(self.FOT4_tensor)
        return self

    def get_eigensystem(self):
        self.get_eigensystem_func = self._select_get_eigensystem_function()
        self.get_eigensystem_func()
        return self

    def _select_get_eigensystem_function(self):
        return getattr(
            self,
            {sym: f"_get_eigensystem_if_FOT2_{sym}" for sym in self._keys_sym_FOT2}[
                self.FOT2_symmetry
            ],
        )

    def _get_eigensystem_if_FOT2_isotropic(self):
        self.decomposer_class = decompositions.DecompositionSelector(
            self.FOT4_mandel6_dev
        ).select()
        self.decomposer = self.decomposer_class(FOT4_deviator=self.FOT4_mandel6_dev)

        self.eigen_vector_which_contains_eigensystem_info = (
            self.decomposer.get_eigen_vector_which_contains_eigensystem_info()
        )

        _, self.eigensystem = np.linalg.eigh(
            converter.to_tensor(self.eigen_vector_which_contains_eigensystem_info)
        )

    def _get_eigensystem_if_FOT2_transversely_isotropic(self):
        pass

    def _get_eigensystem_if_FOT2_orthotropic(self):
        pass

    def _get_deviator(self, mandel):
        tensor = converter.to_tensor(mandel)
        deviator = mechkit.operators.dev(tensor)
        return converter.to_mandel6(deviator)

    def _assert_is_FOT4(self, candidate):
        assert (candidate.shape == (3, 3, 3, 3)) or (candidate.shape == (6, 6))
        assert np.allclose(mechkit.operators.Sym()(candidate), candidate)


class FourthOrderFabricAnalyser:
    def __init__(self):
        return None

    def analyse(self, FOT4):
        # Start
        analysis = self.analysis = FOT4Analysis(FOT4)
        # Contract
        analysis.calc_FOT2()
        # Identify symmetry FOT2
        analysis.get_symmetry_FOT2()

        # Get eigensystem based on FOT2 and FOT4 information
        analysis.calc_FOT4_deviator()
        analysis.get_eigensystem()

        return analysis
