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


limits = (-1 / 15, 2 / 45)


class TestSpectralDecomposititonOfCubicFOT4Deviator:
    @pytest.mark.parametrize(
        "cubic_by_d1",
        np.linspace(*limits, 3),
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


# [-1.33333333e-01 -1.33333333e-01 -1.33333333e-01  1.30104261e-16  2.00000000e-01  2.00000000e-01]
# 4
# [-7.77777778e-02 -7.77777778e-02 -7.77777778e-02  1.96023753e-16  1.16666667e-01  1.16666667e-01]
# 4
# [-2.22222222e-02 -2.22222222e-02 -2.22222222e-02  1.96891115e-16  3.33333333e-02  3.33333333e-02]
# 4
# [-5.00000000e-02 -5.00000000e-02  1.28369537e-16  3.33333333e-02  3.33333333e-02  3.33333333e-02]
# 0
# [-1.33333333e-01 -1.33333333e-01  1.66533454e-16  8.88888889e-02  8.88888889e-02  8.88888889e-02]
# 0
