import mechkit
import numpy as np
import pytest
import scipy
import sympy as sp
import vofotensors
from vofotensors.abc import d1

import mechinterfabric


np.random.seed(1)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


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
        (-1 / 15, 2 / 45, 0, 1 / 45),
        indirect=True,
    )
    def test_FOT2_symmetry_cubic(self, cubic_by_d1):
        analysis = mechinterfabric.FOT4Analysis(FOT4=cubic_by_d1)
        analysis.get_symmetry_FOT2()
        assert analysis.FOT2_symmetry == "isotropic_or_cubic"

    @pytest.mark.parametrize(
        "cubic_by_d1",
        (-1 / 15, 2 / 45, 0, 1 / 45),
        indirect=True,
    )
    def test_get_eigensystem_if_FOT4_is_cubic(
        self,
        cubic_by_d1,
    ):

        FOT4_rotated = mechinterfabric.utils.rotate_fot4_randomly(cubic_by_d1)

        analysis = mechinterfabric.FOT4Analysis(FOT4_rotated)
        analysis.get_eigensystem()
        FOT4_reconstructed = mechinterfabric.utils.rotate_to_mandel(
            analysis.FOT4.tensor, analysis.eigensystem
        )

        assert np.allclose(cubic_by_d1, FOT4_reconstructed)


###################################################################################


def lambdified_parametrization_transv():
    from vofotensors.abc import alpha1, rho1

    return sp.lambdify(
        [alpha1, rho1],
        vofotensors.fabric_tensors.N4s_parametric["transv_isotropic"]["alpha1_rho1"],
    )


test_cases_passing = [
    {"id": "isotropic", "tensor": sp.lambdify([], vofotensors.basic_tensors.N4_iso)()},
    *[
        {"id": id, "tensor": lambdified_parametrization_transv()(**kwargs)}
        for id, kwargs in [
            ("transv variety of FOT, Fig3, purple", {"alpha1": 0, "rho1": -1 / 90}),
            ("transv variety of FOT, Fig3, grey", {"alpha1": 0, "rho1": 1 / 60}),
            ("transv variety of FOT, Fig3, brown", {"alpha1": -1 / 3, "rho1": 3 / 280}),
            ("transv variety of FOT, Fig3, blue", {"alpha1": 2 / 3, "rho1": 1 / 35}),
            ("transv random, in the middle", {"alpha1": 1 / 6, "rho1": 3 / 280}),
            ("transv variety of FOT, Fig3, brown", {"alpha1": -1 / 3, "rho1": 3 / 280}),
            ("transv variety of FOT, Fig3, blue", {"alpha1": 2 / 3, "rho1": 1 / 35}),
            ("transv random, in the middle", {"alpha1": 1 / 6, "rho1": 3 / 280}),
            (
                "transv random, upper right",
                {"alpha1": 3 / 6, "rho1": (1 / 35 + 1 / 60) / 2},
            ),
        ]
    ],
]

test_cases_failing = []


class TestFOT4AnalysisTransv:
    @pytest.mark.parametrize(
        ("fot4_rotated", "fot4_in_eigensystem"),
        (
            pytest.param(
                mechinterfabric.utils.rotate_fot4_randomly(row["tensor"]),
                row["tensor"],
                id=row["id"],
            )
            for row in test_cases_passing
        ),
    )
    def test_get_eigensystem(self, fot4_rotated, fot4_in_eigensystem):

        analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
        analysis.get_eigensystem()
        reconstructed = mechinterfabric.utils.rotate_to_mandel(
            analysis.FOT4.tensor, analysis.eigensystem
        )

        assert np.allclose(reconstructed, fot4_in_eigensystem)

    @pytest.mark.parametrize(
        ("fot4_rotated", "fot4_in_eigensystem"),
        (
            pytest.param(
                mechinterfabric.utils.rotate_fot4_randomly(row["tensor"]),
                row["tensor"],
                id=row["id"],
            )
            for row in test_cases_failing
        ),
    )
    def test_get_eigensystem_failing(self, fot4_rotated, fot4_in_eigensystem):
        with pytest.raises(mechinterfabric.utils.ExceptionMechinterfabric):
            analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
            analysis.get_eigensystem()
            reconstructed = mechinterfabric.utils.rotate_to_mandel(
                analysis.FOT4.tensor, analysis.eigensystem
            )


###################################################################################


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d1, d3

    return sp.lambdify(
        [alpha1, d1, d3],
        vofotensors.fabric_tensors.N4s_parametric["tetragonal"]["alpha1_d1_d3"],
    )


test_cases_passing = [
    *[
        {"id": id, "tensor": lambdified_parametrization()(**kwargs)}
        for id, kwargs in [
            (f"N2-iso d1={d1}, d3={d3}", {"alpha1": 0, "d1": d1, "d3": d3})
            for d1 in np.linspace(-1 / 15, 2 / 45, 4)[
                :-1
            ]  # Avoid edge case which is tetragonal, but is intepreted as cubic
            for d3 in np.linspace(-1 / 15, -d1 / 4.0, 3)
        ]
    ],
]


test_cases_failing = [
    *[
        {"id": id, "tensor": lambdified_parametrization()(**kwargs)}
        for id, kwargs in [
            ("random pos def 01", {"alpha1": 1 / 6, "d1": -0.009, "d3": 0.0243}),
            ("random pos def 02", {"alpha1": 1 / 3, "d1": 0.01, "d3": 0.01}),
            ("random pos def 03", {"alpha1": -1 / 6, "d1": 0.01, "d3": -0.09}),
        ]
    ],
]


class TestFOT4AnalysisTetragonal:
    @pytest.mark.parametrize(
        ("fot4_rotated", "fot4_in_eigensystem"),
        (
            pytest.param(
                mechinterfabric.utils.rotate_fot4_randomly(row["tensor"]),
                row["tensor"],
                id=row["id"],
            )
            for row in test_cases_passing
        ),
    )
    def test_get_eigensystem(self, fot4_rotated, fot4_in_eigensystem):

        print(f"fot4_rotated=\n{fot4_rotated}")
        print(f"fot4_in_eigensystem=\n{fot4_in_eigensystem}")

        analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
        analysis.get_eigensystem()
        reconstructed = mechinterfabric.utils.rotate_to_mandel(
            analysis.FOT4.tensor, analysis.eigensystem
        )
        print(f"reconstructed=\n{reconstructed}")
        assert np.allclose(reconstructed, fot4_in_eigensystem, atol=1e-7)

    @pytest.mark.parametrize(
        ("fot4_rotated", "fot4_in_eigensystem"),
        (
            pytest.param(
                mechinterfabric.utils.rotate_fot4_randomly(row["tensor"]),
                row["tensor"],
                id=row["id"],
            )
            for row in test_cases_failing
        ),
    )
    def test_get_eigensystem_failing(self, fot4_rotated, fot4_in_eigensystem):
        with pytest.raises(mechinterfabric.utils.ExceptionMechinterfabric):
            analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
            analysis.get_eigensystem()
            reconstructed = mechinterfabric.utils.rotate_to_mandel(
                analysis.FOT4.tensor, analysis.eigensystem
            )


###################################################################################


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d3, d9

    return sp.lambdify(
        [alpha1, d3, d9],
        vofotensors.fabric_tensors.N4s_parametric["trigonal"]["alpha1_d3_d9"],
    )


test_cases_passing = [
    *[
        {"id": id, "tensor": lambdified_parametrization()(**kwargs)}
        for id, kwargs in [
            ("random pos def 01", {"alpha1": 0, "d3": 0.0125, "d9": 0.0325}),
            ("random pos def 02", {"alpha1": 1 / 3, "d3": 0.0125, "d9": 0.0325}),
            ("random pos def 02", {"alpha1": -1 / 3, "d3": 0.0125, "d9": 0.0325}),
            ("random pos def 02", {"alpha1": -1 / 3, "d3": 0.0055, "d9": 0.0125}),
        ]
    ],
]


class TestFOT4AnalysisTrigonal:
    @pytest.mark.parametrize(
        ("fot4_rotated", "fot4_in_eigensystem"),
        (
            pytest.param(
                mechinterfabric.utils.rotate_fot4_randomly(row["tensor"]),
                row["tensor"],
                id=row["id"],
            )
            for row in test_cases_passing
        ),
    )
    def test_get_eigensystem(self, fot4_rotated, fot4_in_eigensystem):

        analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
        analysis.get_eigensystem()
        reconstructed = mechinterfabric.utils.rotate_to_mandel(
            analysis.FOT4.tensor, analysis.eigensystem
        )
        print(f"fot4_in_eigensystem={fot4_in_eigensystem}")
        print(f"reconstructed={reconstructed}")
        assert np.allclose(reconstructed, fot4_in_eigensystem)
