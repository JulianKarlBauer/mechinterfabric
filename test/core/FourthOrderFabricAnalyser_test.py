import mechinterfabric
import numpy as np
import pytest
import vofotensors
from vofotensors.abc import d1
import sympy as sp


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


class TestAnalyser:
    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45),
        indirect=True,
    )
    def test_FOT2_symmetry_cubic(self, analyser, cubic_by_d1):
        result = analyser.analyse(FOT4=cubic_by_d1)
        assert result.FOT2_symmetry == "isotropic"
