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
def lambdified_parametrization_cubic():
    parametrization_symbolic = vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"]
    return sp.lambdify([d1], parametrization_symbolic)


@pytest.fixture()
def cubic_by_d1(lambdified_parametrization_cubic, request):
    return lambdified_parametrization_cubic(d1=request.param)


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

    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45),
        indirect=True,
    )
    def test_get_eigensystem_if_FOT4_is_cubic(
        self,
        cubic_by_d1,
    ):

        ######
        # Rotate

        FOT4_rotated = utils.rotate_fot4_randomly(cubic_by_d1)

        analysis = mechinterfabric.FOT4Analysis(FOT4_rotated)
        analysis.get_eigensystem()
        FOT4_reconstructed = utils.converter.to_mandel6(
            mechinterfabric.utils.rotate(analysis.FOT4.tensor, analysis.eigensystem)
        )
        assert np.allclose(cubic_by_d1, FOT4_reconstructed)
