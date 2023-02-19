from collections import Counter

import mechkit
import numpy as np

from . import utils


class SpectralDecompositionFOT2:
    def __init__(self, FOT2):
        self.FOT2 = FOT2

    def get_symmetry(self):
        self._decompose()
        return self._identify_symmetry_FOT2()

    def _decompose(self):
        (
            self.FOT2_eigenvalues,
            self.FOT2_rotation,
        ) = utils.get_eigenvalues_and_rotation_matrix_into_eigensystem(self.FOT2)
        return self

    def _identify_symmetry_FOT2(self):
        index_pairs = [[0, 1], [0, 2], [1, 2]]

        self.FOT2_eigenvalues_are_equal = tuple(
            self._pair_of_eigenvalues_is_equal(*self.FOT2_eigenvalues[pair])
            for pair in index_pairs
        )

        self.FOT2_symmetry = self._map_equal_eigenvalue_pairs_to_symmetry()
        return self.FOT2_symmetry

    def _map_equal_eigenvalue_pairs_to_symmetry(self):
        return {
            (True, True, True): "isotropic",
            (True, True, False): "transversely_isotropic",
            # We assume, that eigenvalues are sorted, so v0 == v2 is only possible, if v0 == v1 as well
            (False, True, True): "transversely_isotropic",
            (False, False, False): "orthotropic",
        }[self.FOT2_eigenvalues_are_equal]

    def _pair_of_eigenvalues_is_equal(self, first, second, atol=1e-4, rtol=1e-4):
        return np.isclose(first, second, atol=atol, rtol=rtol)


class DecompositionSelector:
    def __init__(self, FOT4_deviator=None, decimals_precision=4):
        self.decimals_precision = decimals_precision
        if FOT4_deviator is not None:
            self.eigen_values, self.eigen_vectors = np.linalg.eigh(FOT4_deviator)

    def _get_rounded_eigenvalues(self):
        self.eigen_values_rounded = np.around(
            self.eigen_values, self.decimals_precision
        )

    def _count_eigenvalues_and_create_lookups(self):
        self.counter_eigenvalues = Counter(self.eigen_values_rounded)
        most_common_eigenvalues = self.counter_eigenvalues.most_common()
        self.eigen_values_counted, self.eigen_values_counted_multiplicity = list(
            zip(*most_common_eigenvalues)
        )

        self.eigen_values_indices = [
            np.argwhere(self.eigen_values_rounded == value).flatten()
            for value in self.eigen_values_counted
        ]

    def select(self):
        self._get_rounded_eigenvalues()
        self._count_eigenvalues_and_create_lookups()
        self.symmetry = self._identify_symmetry()
        print(self.symmetry)
        # return self._get_decomposer()

    def _identify_symmetry(self):
        match self.eigen_values_counted_multiplicity:
            case (6,):
                return "isotropic"
            case (3, 2, 1):
                return "cubic"
            case (2, 1, 1, 1, 1):
                return "tetragonal"
            case (2, 2, 1, 1):
                return "trigonal or transversely isotropic"
            case (1, 1, 1, 1, 1, 1):
                return "orthotropic or higher"
            case _:
                raise utils.ExceptionMechinterfabric(
                    "Unknown symmetry class for multiplicity = "
                    + f"{self.eigen_values_counted_multiplicity}"
                )

    def _get_decomposer(self):
        match self.symmetry:
            case "isotropic":
                pass
            case "cubic":
                return SpectralDecomposititonOfCubicFOT4Deviator
            case _:
                raise utils.ExceptionMechinterfabric(
                    "No known decomposer for symmetry = " + f"{self.symmetry}"
                )


class SpectralDecomposititonOfCubicFOT4Deviator:
    def __init__(self, FOT4_deviator=None, decimals_precision=4):
        self.decimals_precision = decimals_precision
        if FOT4_deviator is not None:
            self.eigen_values, self.eigen_vectors = np.linalg.eigh(FOT4_deviator)

    def get_eigen_vector_which_contains_eigensystem_info(
        self,
        select_only_one_vector=True,
    ):
        self._get_rounded_eigenvalues()
        self._get_most_common_eigenvalues()
        self._assert_eigenvalues_are_cubic()
        self._get_index_two_fold_eigenvalue_of_cubic_deviator(
            select_only_one_vector=select_only_one_vector
        )
        self._get_eigen_vector_two_fold_eigen_value()

        return self.eigen_vector_two_fold_eigen_value

    def _get_rounded_eigenvalues(self):
        self.eigen_values_rounded = np.around(
            self.eigen_values, self.decimals_precision
        )  # .tolist()

    def _assert_eigenvalues_are_cubic(self):
        positions_in_most_common_to_be_asserted = {
            0: {
                "repetition": 3,
                "message": "One eigenvalue should occur three times and corresponds to shear eigen mode",
            },
            1: {
                "repetition": 2,
                "message": "One eigenvalue should occur twice and its corresponding eigen-vector contains the eigen-system information",
            },
            2: {
                "repetition": 1,
                "message": "One eigenvalue should occure once and is equal to zero. It corresponds to the isotropic mode which is not contained in the deviator",
            },
        }
        for (
            position,
            details,
        ) in positions_in_most_common_to_be_asserted.items():
            assert (
                self.most_common_eigenvalues[position][1] == details["repetition"]
            ), details["message"]

    def _get_most_common_eigenvalues(self):
        counter = Counter(self.eigen_values_rounded)
        self.most_common_eigenvalues = counter.most_common()

    def _get_index_two_fold_eigenvalue_of_cubic_deviator(self, select_only_one_vector):
        position_of_interest = 1
        eigen_value_of_interest = self.most_common_eigenvalues[position_of_interest][0]
        # self.index_two_fold_eigen_value = self.eigen_values_rounded.tolist().index(
        #     eigen_value_of_interest
        # )
        indices = np.argwhere(
            self.eigen_values_rounded == eigen_value_of_interest
        ).flatten()
        self.index_two_fold_eigen_value = (
            indices[0] if select_only_one_vector else indices
        )

    def _get_eigen_vector_two_fold_eigen_value(self):
        # See structure of eigen vectors
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
        self.eigen_vector_two_fold_eigen_value = self.eigen_vectors[
            :, self.index_two_fold_eigen_value
        ].T
