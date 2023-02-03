import numpy as np
import mechkit
from . import utils

converter = mechkit.notation.Converter()


class FOT4Analysis:
    def __init__(self, FOT4):
        self._assert_is_FOT4(FOT4)
        self.FOT4_mandel6 = converter.to_mandel6(FOT4)

    def calc_FOT2(self):
        self.FOT2 = self._contract_FOT4_to_FOT2(self.FOT4_mandel6)
        return self

    def calc_FOT2_spectral(self):
        (
            self.FOT2_eigenvalues,
            self.FOT2_rotation,
        ) = utils.get_eigenvalues_and_rotation_matrix_into_eigensystem(self.FOT2)
        return self

    def calc_FOT4_deviator(self):
        self.FOT4_mandel6_dev = self._get_deviator(self.FOT4_mandel6)
        return self

    def identify_symmetry_FOT2(self):
        index_pairs = [[0, 1], [0, 2], [1, 2]]
        self.FOT2_eigenvalues_are_equal = tuple(
            self._pair_of_eigenvalues_is_equal(*self.FOT2_eigenvalues[pair])
            for pair in index_pairs
        )
        transv = "transversely_isotropic"

        mapping = {
            (True, True, True): "isotropic",
            (True, True, False): transv,
            # We assume, that eigenvalues are sorted, so v0 == v2 is only possible, if v0 == v1 as well
            (False, True, True): transv,
            (False, False, False): "orthotropic",
        }
        self.FOT2_symmetry = mapping[self.FOT2_eigenvalues_are_equal]
        return self

    def _pair_of_eigenvalues_is_equal(self, first, second, atol=1e-4, rtol=1e-4):
        return np.isclose(first, second, atol=atol, rtol=rtol)

    def _contract_FOT4_to_FOT2(self, mandel):
        I2 = np.eye(3)
        tensor_FOT2 = np.tensordot(converter.to_tensor(mandel), I2)
        return tensor_FOT2

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
        # Contract

        analysis = self.analysis = (
            FOT4Analysis(FOT4).calc_FOT2().calc_FOT2_spectral().calc_FOT4_deviator()
        )

        # Inspect FOT2 part
        print(analysis.FOT2)
        print(analysis.FOT2_eigenvalues)
        print(analysis.FOT2_rotation)
        print(analysis.FOT4_mandel6_dev)

        # Get eigensystem
        # analysis.get_eigensystem()

        # Identify symmetry FOT2
        analysis.identify_symmetry_FOT2()

        analysis.result = {"d0": 0.0, "rotation_Q": np.eye(3)}
        return analysis
