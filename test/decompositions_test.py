import mechkit
import numpy as np
import pytest
import scipy
import sympy as sp
import vofotensors
from vofotensors.abc import d1

import mechinterfabric
from mechinterfabric import utils


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
del f


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
    @pytest.mark.parametrize(
        "eigen_values,first_index_position_two_fold_ev", list_of_tests
    )
    def test__get_index_two_fold_eigenvalue_of_cubic_deviator(
        self, eigen_values, first_index_position_two_fold_ev
    ):

        decomposition = mechinterfabric.decompositions.SpectralDecompositionDeviator4()
        decomposition.eigen_values = eigen_values

        decomposition._get_rounded_eigenvalues()
        decomposition._count_eigenvalues_and_create_lookups()

        locator = mechinterfabric.decompositions.EigensystemLocatorIsotropicCubic(
            decomposition
        )

        locator._get_index_of_eigenvector_which_contains_info_on_eigensystem()
        assert locator.index == first_index_position_two_fold_ev

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
        analysis.get_eigensystem()
        locator = analysis.eigensystem_locator

        eigensystems = [locator.eigensystem]

        locator.index = locator.index + 1
        locator._get_eigenvector_which_contains_info_on_eigensystem()
        locator._calc_eigensystem()

        eigensystems.append(locator.eigensystem)

        # Compare each of the two eigenvectors with initial rotated FOT4
        for eigensystem in eigensystems:
            assert np.allclose(
                cubic_by_d1,
                converter.to_mandel6(utils.rotate(analysis.FOT4.tensor, eigensystem)),
            )
