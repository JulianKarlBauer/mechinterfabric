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
            self.rotation_into_FOT2,
            self.eigenvalus_FOT2,
        ) = utils.get_eigenvalues_and_rotation_matrix_into_eigensystem(self.FOT2)
        return self

    def calc_FOT4_deviator(self):
        self.FOT4_mandel6_dev = self._get_deviator(self.FOT4_mandel6)
        return self

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

        self.analysis = analysis = (
            FOT4Analysis(FOT4).calc_FOT2().calc_FOT2_spectral().calc_FOT4_deviator()
        )

        print(analysis.FOT2)

        # Identify symmetry FOT2
        # Get eigensystem candidates

        print(analysis.rotation_into_FOT2)
        print(analysis.eigenvalus_FOT2)
        print(analysis.FOT4_mandel6_dev)

        analysis.result = {"d0": 0.0, "rotation_Q": np.eye(3)}
        return analysis
