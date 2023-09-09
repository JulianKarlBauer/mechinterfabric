import mechkit
import numpy as np
import pytest
import scipy
import sympy as sp
import vofotensors
from vofotensors.abc import alpha1
from vofotensors.abc import alpha3
from vofotensors.abc import d1
from vofotensors.abc import d2
from vofotensors.abc import d3
from vofotensors.abc import d4
from vofotensors.abc import d5
from vofotensors.abc import d6
from vofotensors.abc import d7
from vofotensors.abc import d8
from vofotensors.abc import d9
from vofotensors.abc import la1
from vofotensors.abc import la2
from vofotensors.abc import rho1

import mechinterfabric


np.random.seed(1)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


def lambdified_parametrization_cubic():
    return sp.lambdify(
        [d1],
        vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"],
    )


def lambdified_parametrization_transv():
    return sp.lambdify(
        [alpha1, rho1],
        vofotensors.fabric_tensors.N4s_parametric["transv_isotropic"]["alpha1_rho1"],
    )


def lambdified_parametrization_tetragonal():
    return sp.lambdify(
        [alpha1, d1, d3],
        vofotensors.fabric_tensors.N4s_parametric["tetragonal"]["alpha1_d1_d3"],
    )


def lambdified_parametrization_trigonal():
    return sp.lambdify(
        [alpha1, d3, d9],
        vofotensors.fabric_tensors.N4s_parametric["trigonal"]["alpha1_d3_d9"],
    )


def lambdified_parametrization_triclinic():
    return sp.lambdify(
        [la1, la2, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "la1_la2_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


def lambdified_parametrization_triclinic_alpha():
    return sp.lambdify(
        [alpha1, alpha3, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "alpha1_alpha3_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


###################################################################################


@pytest.fixture()
def cubic_by_d1(request):
    return lambdified_parametrization_cubic()(d1=request.param)


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


###################################################################################
def limit_d9_trigonal(alpha1, d3):
    return (
        np.sqrt(
            98
            - 105 * alpha1
            - 225 * alpha1**2
            - 4410 * d3
            + 14175 * alpha1 * d3
            - 88200 * d3**2
        )
    ) / (np.sqrt(2) * 105)


test_cases_passing = [
    {"id": "isotropic", "tensor": sp.lambdify([], vofotensors.basic_tensors.N4_iso)()},
    *[
        {"id": id, "tensor": lambdified_parametrization_cubic()(**kwargs)}
        for id, kwargs in [
            *[(f"Cubic d1={d1}", {"d1": d1}) for d1 in (-1 / 15, 2 / 45, 0, 1 / 45)],
        ]
    ],
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
    *[
        {"id": id, "tensor": lambdified_parametrization_tetragonal()(**kwargs)}
        for id, kwargs in [
            *[
                (f"tetra N2-iso d1={d1}, d3={d3}", {"alpha1": 0, "d1": d1, "d3": d3})
                for d1 in np.linspace(-1 / 15, 2 / 45, 4)[
                    :-1
                ]  # Avoid edge case which is tetragonal, but is intepreted as cubic
                for d3 in np.linspace(-1 / 15, -d1 / 4.0, 3)
            ],
            ("tetra pos def 01", {"alpha1": 1 / 6, "d1": -0.009, "d3": -0.019799}),
            ("tetra pos def 02", {"alpha1": 1 / 3, "d1": 0.01, "d3": -0.0179}),
            ("tetra pos def 03", {"alpha1": -1 / 6, "d1": 0.01, "d3": -0.09}),
            ("tetra pos def 04", {"alpha1": 1 / 2, "d1": -0.09, "d3": 0.01}),
        ]
    ],
    *[
        {"id": id, "tensor": lambdified_parametrization_trigonal()(**kwargs)}
        for id, kwargs in [
            *[
                (
                    f"trig alpha={alpha1}, d3={d3}, d9={d9}",
                    {"alpha1": 0, "d3": d3, "d9": d9},
                )
                for alpha1 in np.linspace(-1 / 3, 2 / 3, 3)[:-1]
                for d3 in np.linspace(
                    (-28 - 60 * alpha1 + 315 * alpha1**2) / 2520,
                    (14 + 15 * alpha1) / 840,
                    5,
                )[1:-1]
                for d9 in np.linspace(0, limit_d9_trigonal(alpha1, d3), 5)[1:-1]
            ],
            ("trig pos def 01", {"alpha1": 0, "d3": 0.0125, "d9": 0.0325}),
            ("trig pos def 02", {"alpha1": 1 / 3, "d3": 0.0125, "d9": 0.0325}),
        ]
    ],
    *[
        {
            "id": id,
            "tensor": lambdified_parametrization_triclinic()(
                d4=0, d5=0, d6=0, d7=0, d8=0, d9=0, **kwargs
            ),
        }
        for id, kwargs in [
            (
                "ortho N2-iso 01",
                {"la1": 1 / 3, "la2": 1 / 3, "d1": 0.05, "d2": 0.033, "d3": 0.011},
            ),
            (
                "ortho N2-iso 02",
                {"la1": 1 / 3, "la2": 1 / 3, "d1": 0.03, "d2": 0.02, "d3": 0.01},
            ),
            (
                "ortho N2-transv prolate 01",
                {"la1": 1 / 2, "la2": 1 / 4, "d1": 0.015, "d2": 0.01, "d3": 0.005},
            ),
        ]
    ],
    *[
        {
            "id": id,
            "tensor": lambdified_parametrization_triclinic_alpha()(
                d4=0, d5=0, d6=0, d7=0, d8=0, d9=0, **kwargs
            ),
        }
        for id, kwargs in [
            (
                "ortho N2-transv prolate 02",
                {
                    "alpha1": 1 / 2,
                    "alpha3": 0,
                    "d1": -0.085,
                    "d2": -0.09,
                    "d3": 0.035,
                },
            ),
            (
                "ortho N2-transv oblate 01",
                {
                    "alpha1": 0,
                    "alpha3": -1 / 6,
                    "d1": -0.06,
                    "d2": 0.02,
                    "d3": -0.015,
                },
            ),
            (
                "ortho N2-transv oblate 02",
                {
                    "alpha1": 0,
                    "alpha3": -1 / 4,
                    "d1": -0.05,
                    "d2": -0.01,
                    "d3": -0.02,
                },
            ),
        ]
    ],
    *[
        {
            "id": id,
            "tensor": lambdified_parametrization_triclinic()(**kwargs),
        }
        for id, kwargs in [
            (
                "tricl from random fibers 01",
                {
                    "la1": 0.6504,
                    "la2": 0.23199,
                    "d1": -0.01158,
                    "d2": -0.00754,
                    "d3": -0.00449,
                    "d4": 0.01866,
                    "d5": -0.00673,
                    "d6": 0.02392,
                    "d7": -0.0145,
                    "d8": 0.01395,
                    "d9": -0.01621,
                },
            ),
        ]
    ],
]


test_cases_failing = [
    *[
        {"id": id, "tensor": lambdified_parametrization_trigonal()(**kwargs)}
        for id, kwargs in [
            # ("trig accident cubic", {"alpha1": 0, "d3": -0.01111, "d9": 0.078567420}),
        ]
    ],
]

test_cases_raises = [
    *[
        {"id": id, "tensor": lambdified_parametrization_tetragonal()(**kwargs)}
        for id, kwargs in [
            # Edge cases which are tetragonal, but are intepreted as cubic
            (
                "tetra accidentaly cubic alpha1=1/3",
                {"alpha1": 1 / 3, "d1": 0.01, "d3": 0.01},
            ),
            # (
            #     "tetra accidentaly cubic alpha1=-1/6",
            #     {"alpha1": -1 / 6, "d1": -0.06666, "d3": -0.06666},
            # ),
        ]
    ],
]


class TestFOT4Analysis:
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

        print(f"Min(eigenvalues)={np.min(np.linalg.eigh(fot4_rotated)[0])}")
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
        print(f"Min(eigenvalues)={np.min(np.linalg.eigh(fot4_rotated)[0])}")

        analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
        analysis.get_eigensystem()
        reconstructed = mechinterfabric.utils.rotate_to_mandel(
            analysis.FOT4.tensor, analysis.eigensystem
        )
        assert not np.allclose(reconstructed, fot4_in_eigensystem, atol=1e-7)

    @pytest.mark.parametrize(
        ("fot4_rotated", "fot4_in_eigensystem"),
        (
            pytest.param(
                mechinterfabric.utils.rotate_fot4_randomly(row["tensor"]),
                row["tensor"],
                id=row["id"],
            )
            for row in test_cases_raises
        ),
    )
    def test_get_eigensystem_raises(self, fot4_rotated, fot4_in_eigensystem):
        print(f"Min(eigenvalues)={np.min(np.linalg.eigh(fot4_rotated)[0])}")

        with pytest.raises(mechinterfabric.utils.ExceptionMechinterfabric):
            analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
            analysis.get_eigensystem()
