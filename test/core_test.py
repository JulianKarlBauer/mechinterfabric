import mechkit
import numpy as np
import pytest
import scipy
import sympy as sp
import vofotensors
from vofotensors.abc import d1

import mechinterfabric
from . import utils

np.random.seed(1)


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
    utils.converter = mechkit.notation.Converter()

    def rotate(mandel):

        return utils.converter.to_mandel6(
            mechinterfabric.utils.rotate(utils.converter.to_tensor(mandel), Q=Q)
        )

    return rotate


class TestFOT4AnalysisCubic:
    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45),
        indirect=True,
    )
    def test_FOT2_symmetry_cubic(self, cubic_by_d1):
        analysis = mechinterfabric.FOT4Analysis(FOT4=cubic_by_d1)
        analysis.get_symmetry_FOT2()
        assert analysis.FOT2_symmetry == "isotropic"

    @pytest.mark.parametrize("angle", [17, 211, 52, 360 * np.random.rand()])
    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45),
        indirect=True,
    )
    def test__get_eigensystem_if_FOT4_is_cubic(
        self,
        cubic_by_d1,
        angle,
    ):

        ######
        # Rotate

        rotate = get_rotating_func(angle_in_degree=angle)
        FOT4_rotated = rotate(cubic_by_d1)

        analysis = mechinterfabric.FOT4Analysis(FOT4_rotated)
        analysis.get_eigensystem()
        FOT4_reconstructed = utils.converter.to_mandel6(
            mechinterfabric.utils.rotate(analysis.FOT4.tensor, analysis.eigensystem)
        )
        assert np.allclose(cubic_by_d1, FOT4_reconstructed)
