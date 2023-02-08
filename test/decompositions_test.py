import mechinterfabric
from mechinterfabric import utils
import numpy as np
import pytest
import vofotensors
from vofotensors.abc import d1
import sympy as sp
import scipy
import mechkit


@pytest.fixture()
def converter():
    return mechkit.notation.Converter()


@pytest.fixture()
def lambdified_parametrization_cubic(request):
    parametrization_symbolic = vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"]
    return sp.lambdify([d1], parametrization_symbolic)


@pytest.fixture()
def cubic_by_d1(lambdified_parametrization_cubic, request):
    return lambdified_parametrization_cubic(d1=request.param)


f = np.array
list_of_tests = [
    (f([-1.3333e-01, -1.3333e-01, -1.3333e-01, 1.3010e-16, 2.0000e-01, 2.0000e-01]), 4),
    (f([-7.7777e-02, -7.7777e-02, -7.7777e-02, 1.9602e-16, 1.1666e-01, 1.1666e-01]), 4),
    (f([-2.2222e-02, -2.2222e-02, -2.2222e-02, 1.9689e-16, 3.3333e-02, 3.3333e-02]), 4),
    (f([-5.0000e-02, -5.0000e-02, 1.2836e-16, 3.3333e-02, 3.3333e-02, 3.3333e-02]), 0),
    (f([-1.3333e-01, -1.3333e-01, 1.6653e-16, 8.8888e-02, 8.8888e-02, 8.8888e-02]), 0),
]


@pytest.fixture()
def func_rotating_FOT4(converter):
    def func(tensor_or_mandel):

        ######
        # Rotate
        angle = 52
        rotation_vector = np.array([0, 0, 1])

        rotation = scipy.spatial.transform.Rotation.from_rotvec(
            angle * rotation_vector, degrees=True
        )
        Q = rotation.as_matrix()

        def rotate(mandel, Q):
            return converter.to_mandel6(
                mechinterfabric.utils.rotate(converter.to_tensor(mandel), Q=Q)
            )

        return rotate(tensor_or_mandel, Q=Q)

    return func


class TestSpectralDecomposititonOfCubicFOT4Deviator:
    @pytest.mark.parametrize("eigen_values,first_index", list_of_tests)
    def test__get_index_two_fold_eigenvalue_of_cubic_deviator(
        self, eigen_values, first_index
    ):

        decomposition = (
            mechinterfabric.decompositions.SpectralDecomposititonOfCubicFOT4Deviator()
        )
        decomposition.eigen_values = eigen_values
        decomposition._get_rounded_eigenvalues()
        decomposition._get_most_common_eigenvalues()
        decomposition._get_index_two_fold_eigenvalue_of_cubic_deviator(
            select_only_one_vector=True  # Get only first index
        )
        assert decomposition.index_two_fold_eigen_value == first_index

    @pytest.mark.parametrize(
        "cubic_by_d1",
        [-1 / 15, 2 / 45],
        indirect=True,
    )
    def test_eigenvectors_of_both_two_fold_eigen_values_lead_to_similar_rotation(
        self,
        cubic_by_d1,
        func_rotating_FOT4,
        converter,
    ):
        rotated = func_rotating_FOT4(cubic_by_d1)
        analysis = mechinterfabric.FOT4Analysis(FOT4=rotated)
        analysis.calc_FOT4_deviator()

        analysis.spectral_decomp_FOT_dev = (
            mechinterfabric.decompositions.SpectralDecomposititonOfCubicFOT4Deviator(
                FOT4_deviator=analysis.FOT4_mandel6_dev
            )
        )
        analysis.eigen_vector_which_contains_eigensystem_info = analysis.spectral_decomp_FOT_dev.get_eigen_vector_which_contains_eigensystem_info(
            select_only_one_vector=False  # This triggers returning of both eigen vectors which correspond to two-fold eigen value
        )

        # Compare each of the two eigenvectors with initial rotated FOT4
        for vector in analysis.eigen_vector_which_contains_eigensystem_info:
            print("vector=", vector)
            _, rotation = np.linalg.eigh(converter.to_tensor(vector))
            assert np.allclose(
                cubic_by_d1,
                converter.to_mandel6(utils.rotate(analysis.FOT4_tensor, rotation)),
            )
