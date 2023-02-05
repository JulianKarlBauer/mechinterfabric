import numpy as np
import mechkit
from . import utils

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

    def calc_FOT2_spectral(self):
        (
            self.FOT2_eigenvalues,
            self.FOT2_rotation,
        ) = utils.get_eigenvalues_and_rotation_matrix_into_eigensystem(self.FOT2)
        return self

    def calc_FOT4_deviator(self):
        self.FOT4_mandel6_dev = self._get_deviator(self.FOT4_tensor)
        return self

    def identify_symmetry_FOT2(self):
        index_pairs = [[0, 1], [0, 2], [1, 2]]

        self.FOT2_eigenvalues_are_equal = tuple(
            self._pair_of_eigenvalues_is_equal(*self.FOT2_eigenvalues[pair])
            for pair in index_pairs
        )

        self.FOT2_symmetry = self._map_equal_eigenvalue_pairs_to_symmetry()
        return self

    def _map_equal_eigenvalue_pairs_to_symmetry(self):
        return {
            (True, True, True): "isotropic",
            (True, True, False): "transversely_isotropic",
            # We assume, that eigenvalues are sorted, so v0 == v2 is only possible, if v0 == v1 as well
            (False, True, True): "transversely_isotropic",
            (False, False, False): "orthotropic",
        }[self.FOT2_eigenvalues_are_equal]

    def get_eigensystem(self):
        self.get_eigensystem_func = self._select_get_eigensystem_function()
        self.get_eigensystem_func()
        return self

    def _select_get_eigensystem_function(self):
        return getattr(
            self,
            {sym: f"_get_eigensystem_FOT2_{sym}" for sym in self._keys_sym_FOT2}[
                self.FOT2_symmetry
            ],
        )

    def _get_eigensystem_FOT2_isotropic(self):
        pass

    def _get_eigensystem_FOT2_transversely_isotropic(self):
        pass

    def _get_eigensystem_FOT2_orthotropic(self):
        pass

    def _pair_of_eigenvalues_is_equal(self, first, second, atol=1e-4, rtol=1e-4):
        return np.isclose(first, second, atol=atol, rtol=rtol)

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
        analysis.calc_FOT2_spectral()
        # Identify symmetry FOT2
        analysis.identify_symmetry_FOT2()

        # Get eigensystem based on FOT2 and FOT4 information
        analysis.calc_FOT4_deviator()
        analysis.get_eigensystem()

        return analysis
