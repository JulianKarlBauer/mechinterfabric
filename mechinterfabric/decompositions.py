from collections import Counter

import mechkit
import numpy as np

from . import utils

converter = mechkit.notation.Converter()


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
        ) = utils.get_eigenvalues_and_rotation_matrix_into_eigensystem(self.FOT2.tensor)
        return self

    def _identify_symmetry_FOT2(self):
        index_pairs = [[0, 1], [1, 2]]

        self.FOT2_eigenvalues_are_equal = tuple(
            self._pair_of_eigenvalues_is_equal(*self.FOT2_eigenvalues[pair])
            for pair in index_pairs
        )

        self.FOT2_symmetry = self._map_equal_eigenvalue_pairs_to_symmetry()
        return self.FOT2_symmetry

    def _map_equal_eigenvalue_pairs_to_symmetry(self):
        # We assume, that eigenvalues are sorted
        return {
            (True, True): "isotropic_or_cubic",
            # Oblate
            (True, False): "transversely_isotropic_or_tetragonal_or_trigonal",
            # Prolate
            (False, True): "transversely_isotropic_or_tetragonal_or_trigonal",
            (False, False): "orthotropic_or_monoclinic_or_triclinic",
        }[self.FOT2_eigenvalues_are_equal]

    def _pair_of_eigenvalues_is_equal(self, first, second, atol=1e-4, rtol=1e-4):
        return np.isclose(first, second, atol=atol, rtol=rtol)


class SpectralDecompositionDeviator4:
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
        self.eigenvalues_most_common = self.counter_eigenvalues.most_common()
        self.eigen_values_counted, self.eigen_values_counted_multiplicity = list(
            zip(*self.eigenvalues_most_common)
        )

        self.eigen_values_indices = [
            np.argwhere(self.eigen_values_rounded == value).flatten()
            for value in self.eigen_values_counted
        ]

    def get_symmetry(self):
        self._get_rounded_eigenvalues()
        self._count_eigenvalues_and_create_lookups()
        self.symmetry = self._identify_symmetry()
        return self.symmetry

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


class EigensystemLocator:
    def __init__(self, spectral_decomposition):
        self.spectral_decomposition = spectral_decomposition


class EigensystemLocatorIsotropicIsotropic(EigensystemLocator):
    def get_eigensystem(self):
        return np.eye(3)


class EigensystemLocatorIsotropicCubic(EigensystemLocator):
    def __init__(self, spectral_decomposition):
        super().__init__(spectral_decomposition)
        self._assert_eigenvalues_are_cubic()

    def get_eigensystem(self):
        self._get_index_of_eigenvector_which_contains_info_on_eigensystem()
        self._get_eigenvector_which_contains_info_on_eigensystem()
        self._calc_eigensystem()
        return self.eigensystem

    def _get_index_of_eigenvector_which_contains_info_on_eigensystem(self):
        assert self.spectral_decomposition.eigen_values_counted_multiplicity[1] == 2
        self.index = self.spectral_decomposition.eigen_values_indices[1][0]

    def _get_eigenvector_which_contains_info_on_eigensystem(self):
        self.eigen_vector_two_fold_eigen_value = (
            self.spectral_decomposition.eigen_vectors[:, self.index].T
            # See structure of eigen vectors
            # https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
        )

    def _calc_eigensystem(self):
        _, self.eigensystem = np.linalg.eigh(
            converter.to_tensor(self.eigen_vector_two_fold_eigen_value)
        )

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
                self.spectral_decomposition.eigenvalues_most_common[position][1]
                == details["repetition"]
            ), details["message"]


class EigensystemLocatorTransvTetra(EigensystemLocator):
    def get_eigensystem(self):
        self.eigensystem = self.get_eigenvec_with_eigenvalues_m211()
        return self.eigensystem

    def get_eigenvec_with_eigenvalues_m211(self, tol=1e-3):
        def allclose(A, B):
            return np.allclose(A, B, rtol=tol, atol=tol)

        factor = 1.0 / np.sqrt(6)
        for vector in self.spectral_decomposition.eigen_vectors.T:
            tensor = converter.to_tensor(vector)
            vals, vecs = np.linalg.eigh(tensor)

            # Sign of eigenvectors are arbitrary, we expect a specific sign convention,
            # see variable "reference"
            # Start with sorting both vals and vecs by increasing absolute values of vals
            index = np.argsort(np.abs(vals))
            vals = vals[index]
            vecs = vecs[:, index]
            if vals[-1] <= 0:
                vals = -vals

            reference = 1.0 / np.sqrt(6) * np.array([-1.0, -1.0, 2.0])

            if allclose(vals, reference):
                vals_sorted, eigensystem = utils.sort_eigen_values_and_vectors(
                    eigen_values=vals, eigen_vectors=vecs
                )
                print(f"vals = {vals}")
                print(f"vals_sorted = {vals_sorted}")
                return eigensystem
        raise utils.ExceptionMechinterfabric(
            "None of the eigenvalue triplets matched the reference"
        )
