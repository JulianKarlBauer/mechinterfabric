from pprint import pprint

import mechkit
import numpy as np
import scipy
import sympy as sp
import vofotensors

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=4)


con = mechkit.notation.Converter()


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d1, d3

    return sp.lambdify(
        [alpha1, d1, d3],
        vofotensors.fabric_tensors.N4s_parametric["tetragonal"]["alpha1_d1_d3"],
    )


# {"alpha1": 0, "d1": 0.025, "d3": 0.013}


def allclose(A, B):
    tol = 1e-5
    return np.allclose(A, B, rtol=tol, atol=tol)


def handle_near_zero_negatives(value):
    # Catch problem with very small negative numbers
    if np.isclose(value, 0.0):
        value = 0.0
    return value


def run(d1, d3):
    example = {"alpha1": 0, "d1": d1, "d3": d3}
    tol = 5
    print(f"d1={np.round(d1, tol)}\t d3={np.round(d3, tol)}")

    fot4 = lambdified_parametrization()(**example)
    tmp = handle_near_zero_negatives(np.min(np.linalg.eigh(fot4)[0]))
    assert tmp >= 0, "has to be positive semi definite"

    deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))

    # random_rotation = mechinterfabric.utils.get_rotation_by_vector(
    #     vector=np.random.rand() * 360 * np.array([1, 0, 0]), degrees=True
    # )
    random_rotation = mechinterfabric.utils.get_random_rotation()
    disturbed = mechinterfabric.utils.rotate_to_mandel(
        deviator,
        Q=random_rotation,
    )
    disturbed_fot4 = mechinterfabric.utils.rotate_to_mandel(
        fot4,
        Q=random_rotation,
    )
    analysis = mechinterfabric.FOT4Analysis(disturbed_fot4)
    analysis.get_eigensystem()
    transform_tmp = analysis.eigensystem
    candidate = mechinterfabric.utils.rotate_to_mandel(disturbed, transform_tmp)

    def calc_residuum(angle):
        rotation = mechinterfabric.utils.get_rotation_by_vector(
            vector=angle * np.array([1, 0, 0]), degrees=True
        )
        rotated = mechinterfabric.utils.rotate_to_mandel(candidate, Q=rotation)
        indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
        return np.linalg.norm(rotated[indices])

    # Brute force try some angles
    angles = np.linspace(0, 90, 600)
    angles_difference = angles[1] - angles[0]
    residuum = np.zeros((len(angles)), dtype=np.float64)
    for index, angle in enumerate(angles):
        residuum[index] = calc_residuum(angle=angle)

    best_index = np.argmin(residuum)

    solution = scipy.optimize.minimize_scalar(
        calc_residuum,
        bounds=(
            angles[best_index] - angles_difference,
            angles[best_index] + angles_difference,
        ),
        method="bounded",
    )

    optimized_angle = solution.x
    additional_rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=optimized_angle * np.array([1, 0, 0]), degrees=True
    )
    deviator_optimized = mechinterfabric.utils.rotate_to_mandel(
        candidate,
        Q=additional_rotation,
    )

    deviator_alternative = mechinterfabric.utils.rotate_to_mandel(
        deviator_optimized,
        Q=mechinterfabric.utils.get_rotation_by_vector(
            vector=45 * np.array([1, 0, 0]), degrees=True
        ),
    )

    # print(f"deviator=\n{deviator}")
    # print(f"deviator_optimized=\n{deviator_optimized}")
    # print(f"deviator_alternative=\n{deviator_alternative}")

    assert allclose(deviator, deviator_optimized) or allclose(
        deviator, deviator_alternative
    )
    index = np.s_[0, 2]
    d1 = deviator_optimized[index]
    assert np.isclose(d1, deviator_alternative[index])
    index = np.s_[1, 2]
    m1 = deviator_optimized[index]
    m2 = deviator_alternative[index]

    tmp = handle_near_zero_negatives(d1**2 / 16 - m1 * m2)
    summand = np.sqrt(tmp)
    d3s = {"plus": -d1 / 4.0 + summand, "minus": -d1 / 4.0 - summand}
    for key, d3 in d3s.items():
        print(f"d3_{key}   \t={d3}")

    news = {
        key: lambdified_parametrization()(
            alpha1=example["alpha1"], d1=example["d1"], d3=d3
        )
        for key, d3 in d3s.items()
    }

    for key, new in news.items():
        tmp = handle_near_zero_negatives(np.min(np.linalg.eigh(new)[0]))
        assert tmp >= 0
    print("Did assert both deviators: Both fine")
    print()

    from matplotlib import pyplot as plt

    plt.plot(angles, residuum)


# d1s = np.linspace(-1 / 15, 2 / 45, 3)
# for d1 in d1s:
#     print("################################")
#     d3s = np.linspace(-1 / 15, (14 - 105 * d1) / 210, 5)
#     for d3 in d3s:
#         run(d1, d3)

# run(**{"d1": 0.025, "d3": 0.025})
run(**{"d1": 0.044444444444444446, "d3": -0.06666666666666667})
