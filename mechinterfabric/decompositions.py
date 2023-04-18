from collections import Counter

import mechkit
import numpy as np
import scipy

from . import utils

converter = mechkit.notation.Converter()


class SpectralDecompositionFOT2:
    def __init__(self, FOT2):
        self.FOT2 = FOT2

        self.transv_eigenvalue_pair_to_type = {
            (True, False): "oblate",
            (False, True): "prolate",
        }
        self.eigenvalue_pair_to_symmetry = {
            **{
                (True, True): "isotropic_or_cubic",
                (False, False): "orthotropic_or_monoclinic_or_triclinic",
            },
            **{
                key: "transversely_isotropic_or_tetragonal_or_trigonal"
                for key in self.transv_eigenvalue_pair_to_type.keys()
            },
        }

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
        return self.eigenvalue_pair_to_symmetry[self.FOT2_eigenvalues_are_equal]

    def _pair_of_eigenvalues_is_equal(self, first, second, atol=1e-4, rtol=1e-4):
        return np.isclose(first, second, atol=atol, rtol=rtol)

    def _map_equal_eigenvalue_pairs_to_type_of_transversely_isotropy(self):
        return self.transv_eigenvalue_pair_to_type[self.FOT2_eigenvalues_are_equal]


class SpectralDecompositionDeviator4:
    def __init__(self, FOT4_deviator=None, decimals_precision=4):
        self.decimals_precision = decimals_precision
        if FOT4_deviator is not None:
            self.eigen_values, self.eigen_vectors = np.linalg.eigh(FOT4_deviator)
        self.deviator = FOT4_deviator

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
    def __init__(self, FOT4_spectral_decomposition, FOT2_spectral_decomposition):
        self.FOT4_spectral_decomposition = FOT4_spectral_decomposition
        self.FOT2_spectral_decomposition = FOT2_spectral_decomposition

    def _calc_residuum(self, angle):
        rotated = self._rotate_deviator_from_eigensystem_by_angle(angle)
        return self._calc_norm_upper_right_quadrant(rotated)

    def _calc_norm_upper_right_quadrant(self, mandel):
        indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
        return np.linalg.norm(mandel[indices])

    def _rotate_deviator_from_eigensystem_by_angle(self, angle):
        rotation = utils.get_rotation_by_vector(
            vector=angle * np.array([1, 0, 0]), degrees=True
        )
        rotated = utils.rotate_to_mandel(self._deviator_in_eigensystem, Q=rotation)
        return rotated

    def _get_optimal_angle(self, range_angles=[0, 90]):
        # Brute force try some angles

        # Angles optimal for trigonal
        # angles = np.linspace(0, 60, 180)

        # Angles necessary for tetragonal, acceptable for trigonal
        # angles = np.linspace(0, 90, 270)
        angles = np.linspace(*range_angles, range_angles[1] * 3)

        angles_difference = angles[1] - angles[0]
        residuum = np.zeros((len(angles)), dtype=np.float64)
        for index, angle in enumerate(angles):
            residuum[index] = self._calc_residuum(angle=angle)

        best_index = np.argmin(residuum)

        solution = scipy.optimize.minimize_scalar(
            self._calc_residuum,
            bounds=(
                angles[best_index] - angles_difference,
                angles[best_index] + angles_difference,
            ),
            method="bounded",
        )

        optimized_angle = solution.x
        return optimized_angle


class EigensystemLocatorIsotropicIsotropic(EigensystemLocator):
    def get_eigensystem(self, **kwargs):
        return np.eye(3)


class EigensystemLocatorIsotropicCubic(EigensystemLocator):
    def __init__(self, FOT4_spectral_decomposition, FOT2_spectral_decomposition):
        super().__init__(FOT4_spectral_decomposition, FOT2_spectral_decomposition)
        self._assert_eigenvalues_are_cubic()

    def get_eigensystem(self, **kwargs):
        self._get_index_of_eigenvector_which_contains_info_on_eigensystem()
        self._get_eigenvector_which_contains_info_on_eigensystem()
        self._calc_eigensystem()
        return self.eigensystem

    def _get_index_of_eigenvector_which_contains_info_on_eigensystem(self):
        assert (
            self.FOT4_spectral_decomposition.eigen_values_counted_multiplicity[1] == 2
        )
        self.index = self.FOT4_spectral_decomposition.eigen_values_indices[1][0]

    def _get_eigenvector_which_contains_info_on_eigensystem(self):
        self.eigen_vector_two_fold_eigen_value = (
            self.FOT4_spectral_decomposition.eigen_vectors[:, self.index].T
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
                self.FOT4_spectral_decomposition.eigenvalues_most_common[position][1]
                == details["repetition"]
            ), details["message"]


class EigensystemLocatorTransvTrigo(EigensystemLocator):
    def get_eigensystem(self, optimize_rotation_around_x_axis=False, **kwargs):

        # Always make first step
        self.eigensystem = self.get_eigenvec_with_specific_eigenvalues()

        # Prepare check for additional step
        self._deviator_in_eigensystem = utils.rotate_to_mandel(
            self.FOT4_spectral_decomposition.deviator, Q=self.eigensystem
        )
        # Check for second step: Optimize rotation around x-axis
        if (
            optimize_rotation_around_x_axis
            or self.deviator_is_trigonal_or_less_symmetric()
        ):
            # Make additional step for trigonal case
            additional_rotation = self.rotate_into_trigonal_natural_system()

            self.eigensystem = utils.chain_rotations(
                old=self.eigensystem, new=additional_rotation
            )

        return self.eigensystem

    def deviator_is_trigonal_or_less_symmetric(self, tol=1e-6):
        indices_zeros_orthotropic = np.s_[:3, 3:]
        upper_right_quadrant = self._deviator_in_eigensystem[indices_zeros_orthotropic]
        return not np.allclose(
            upper_right_quadrant,
            np.zeros_like(upper_right_quadrant),
            atol=tol,
            rtol=tol,
        )

    def get_eigenvec_with_specific_eigenvalues(self, tol=1e-3):
        def allclose(A, B):
            return np.allclose(A, B, rtol=tol, atol=tol)

        factor = 1.0 / np.sqrt(6)
        for vector in self.FOT4_spectral_decomposition.eigen_vectors.T:
            tensor = converter.to_tensor(vector)
            vals, vecs = np.linalg.eigh(tensor)

            (
                vals,
                vecs,
            ) = self.cast_to_sign_order_convention_of_reference(vals, vecs)

            reference = 1.0 / np.sqrt(6) * np.array([-1.0, -1.0, 2.0])

            if allclose(vals, reference):
                vals_sorted, eigensystem = utils.sort_eigen_values_and_vectors(
                    eigen_values=vals, eigen_vectors=vecs
                )

                return eigensystem
        raise utils.ExceptionMechinterfabric(
            "None of the eigenvalue triplets matched the reference"
        )

    def cast_to_sign_order_convention_of_reference(self, vals, vecs):
        # Sign of eigenvectors are arbitrary, we expect a specific sign convention,
        # see variable "reference"
        # Start with sorting both vals and vecs by increasing absolute values of vals
        index = np.argsort(np.abs(vals))
        vals = vals[index]
        vecs = vecs[:, index]
        if vals[-1] <= 0:
            vals = -vals
        return vals, vecs

    def rotate_into_trigonal_natural_system(self):
        optimized_angle = self._get_optimal_angle()

        additional_rotation = utils.get_rotation_by_vector(
            vector=optimized_angle * np.array([1, 0, 0]), degrees=True
        )
        deviator_optimized = utils.rotate_to_mandel(
            self._deviator_in_eigensystem,
            Q=additional_rotation,
        )

        # If necessary, apply orthogonal transform which changes signs
        # of specific column of off-orthogonal part
        index_positive_value = np.s_[1, 5]
        if deviator_optimized[index_positive_value] <= 0.0:
            transform = np.array(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
            )

            additional_rotation = transform @ additional_rotation

        return additional_rotation


class EigensystemLocatorTetra(EigensystemLocatorTransvTrigo):
    def get_eigensystem(self, **kwargs):

        # Make first two steps
        eigensystem_transformation = super().get_eigensystem(
            optimize_rotation_around_x_axis=True, **kwargs
        )

        # Make third step
        deviator = self.FOT4_spectral_decomposition.deviator
        candidate = utils.rotate_to_mandel(deviator, eigensystem_transformation)

        index_d1 = np.s_[0, 2]
        index_d3_first_dandidate = np.s_[1, 2]

        d1 = candidate[index_d1]
        d3 = candidate[index_d3_first_dandidate]
        transform = np.eye(3)
        print(f"d3 = {d3}")

        center_of_d3_range = -d1 / 4.0
        if center_of_d3_range < d3:
            d3 = 2.0 * center_of_d3_range - d3
            rotation_45_degrees = utils.get_rotation_by_vector(
                vector=45 * np.array([1, 0, 0]), degrees=True
            )
            transform = rotation_45_degrees

            # Test
            print(f"d3 = {d3}")
            assert np.isclose(
                utils.rotate_to_mandel(candidate, transform)[index_d3_first_dandidate],
                d3,
            )

        self.eigensystem = utils.chain_rotations(
            old=eigensystem_transformation, new=transform
        )

        return self.eigensystem


class EigensystemLocatorIsotropicOrthotropicHigher(EigensystemLocator):
    def get_eigensystem(self, **kwargs):

        for value, vector in zip(
            self.FOT4_spectral_decomposition.eigen_values,
            self.FOT4_spectral_decomposition.eigen_vectors.T,
        ):
            if not np.isclose(value, 0.0):
                tensor = converter.to_tensor(vector)
                vals, vecs = np.linalg.eigh(tensor)

                one_over_sqrt_two = 1 / np.sqrt(2)
                if not np.allclose(vals, [-one_over_sqrt_two, 0, one_over_sqrt_two]):

                    (vals_sorted, eigensystem,) = utils.sort_eigen_values_and_vectors(
                        eigen_values=np.abs(vals), eigen_vectors=vecs
                    )
                    deviator_in_unordered_eigensystem = utils.rotate_to_mandel(
                        self.FOT4_spectral_decomposition.deviator, Q=eigensystem
                    )

                    triplet = deviator_in_unordered_eigensystem[[0, 0, 1], [1, 2, 2]]
                    order = np.argsort(triplet)[::-1]

                    (vals_transform, transform,) = utils.sort_eigen_values_and_vectors(
                        eigen_values=triplet[order], eigen_vectors=eigensystem[:, order]
                    )

                    return transform
