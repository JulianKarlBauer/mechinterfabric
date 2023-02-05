import numpy as np
from collections import Counter
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
        eigen_values, eigen_vectors = np.linalg.eigh(self.FOT4_mandel6_dev)
        tmp = self._assert_FOT4_dev_eigenvalues_are_cubic(eigen_values=eigen_values)

        print(eigen_values)
        print(tmp)
        # raise Exception()

    def _get_index_two_fold_eigenvalue_of_cubic_deviator(self):
        pass

    def _assert_FOT4_dev_eigenvalues_are_cubic(
        self, eigen_values, decimals_precision=4
    ):
        eigen_values_rounded = np.around(eigen_values, decimals_precision)
        counter = Counter(eigen_values_rounded)
        assert len(counter) == 3  # Three distinct eigenvalues
        most_common = counter.most_common()
        position_in_most_common_to_number_of_repitition = {
            0: 3,  # One eigenvalue occurs three times and corresponds to shear eigen mode
            1: 2,  # One eigenvalue accours twice and its corresponding eigen-vector contains the eigen-system information
            2: 1,  # One eigenvalue occurs once and is equal to zero. It corresponds to the isotropic mode which is not contained in the deviator
        }
        for (
            position,
            repetition,
        ) in position_in_most_common_to_number_of_repitition.items():
            assert most_common[position][1] == repetition

        position_of_interest = 1
        eigen_value_of_interest = most_common[position_of_interest][0]
        index_eigen_value_of_interest = eigen_values_rounded.tolist().index(
            eigen_value_of_interest
        )
        return index_eigen_value_of_interest

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
