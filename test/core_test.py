import mechinterfabric
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
def analyser():
    return mechinterfabric.FourthOrderFabricAnalyser()


@pytest.fixture()
def lambdified_parametrization_cubic(request):
    parametrization_symbolic = vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"]
    return sp.lambdify([d1], parametrization_symbolic)


@pytest.fixture()
def cubic_by_d1(lambdified_parametrization_cubic, request):
    return lambdified_parametrization_cubic(d1=request.param)


def get_rotating_func(angle_in_degree, vector=None):
    rotation_vector = np.array([0, 0, 1]) if vector is None else np.array(vector)

    rotation = scipy.spatial.transform.Rotation.from_rotvec(
        angle_in_degree * rotation_vector, degrees=True
    )
    Q = rotation.as_matrix()
    converter = mechkit.notation.Converter()

    def rotate(mandel):

        return converter.to_mandel6(
            mechinterfabric.utils.rotate(converter.to_tensor(mandel), Q=Q)
        )

    return rotate


class TestAnalyser:
    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45),
        indirect=True,
    )
    def test_FOT2_symmetry_cubic(self, analyser, cubic_by_d1):
        anaylsis = analyser.analyse(FOT4=cubic_by_d1)
        assert anaylsis.FOT2_symmetry == "isotropic"

    @pytest.mark.parametrize("angle", [17, 211, 52])
    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45),
        indirect=True,
    )
    def test__get_eigensystem_if_FOT4_is_cubic(
        self,
        analyser,
        cubic_by_d1,
        angle,
        converter,
    ):

        ######
        # Rotate

        rotate = get_rotating_func(angle_in_degree=angle)
        FOT4_rotated = rotate(cubic_by_d1)

        analysis = analyser.analyse(FOT4_rotated)
        FOT4_reconstructed = converter.to_mandel6(
            mechinterfabric.utils.rotate(analysis.FOT4_tensor, analysis.eigensystem)
        )
        assert np.allclose(cubic_by_d1, FOT4_reconstructed)