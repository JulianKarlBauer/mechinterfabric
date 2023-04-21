import os
from pprint import pprint

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import symbolic as sb
import sympy as sp
import vofotensors
from symbolic.abc import alpha
from symbolic.numbers import one
from symbolic.numbers import z
from vofotensors.abc import d1

import mechinterfabric

N4 = vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"]

pprint(N4)

#########################
# Parameter limits

# Mathematica

# matrix = {{1/5 - 2 d1,  d1 + 1/15,  d1 + 1/15,           0,           0,           0},
# { d1 + 1/15, 1/5 - 2 d1,  d1 + 1/15,           0,           0,           0},
# { d1 + 1/15,  d1 + 1/15, 1/5 - 2 d1,           0,           0,           0},
# {         0,          0,          0, 2 d1 + 2/15,           0,           0},
# {         0,          0,          0,           0, 2 d1 + 2/15,           0},
# {         0,          0,          0,           0,           0, 2 d1 + 2/15}};

# Result: -1/15 <= d1 <= 2/45

substitutions = ["-1/15", "2/45"]

for expr in substitutions:
    print()
    print("d1=", expr)
    pprint(N4.subs({d1: sp.sympify(expr)}))

w = sp.symbols("w", real=True)

low = N4.subs({d1: sp.sympify(substitutions[0])})
high = N4.subs({d1: sp.sympify(substitutions[1])})

combined = w * low + (sp.S(1) - w) * high

#########################
# Rotate

# Q_around_z = np.array(
#     (
#         (sp.cos(alpha), -sp.sin(alpha), z),
#         (sp.sin(alpha), sp.cos(alpha), z),
#         (z, z, one),
#     ),
# )
#
# N4_rotated = sp.Matrix(sp.trigsimp(sb.actively_rotate_mandel(mandel=N4, Q=Q_around_z)))
#
# pprint(N4_rotated)
#
# I2 = sb.get_I2()
#
# N2 = sb.mandel(np.tensordot(sb.tensorr(N4), sb.to_numpy(sb.get_I2())))

#########################
# Plot
converter = mechkit.notation.Converter()


def sym_to_num_tensor(N4):
    return converter.to_tensor(np.array(N4, dtype=np.float64))


low_num = sym_to_num_tensor(low)
high_num = sym_to_num_tensor(high)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_func_key = "cos_fodf"

mechinterfabric.visualization.plot_stepwise_interpolation_N4_along_x(
    ax=ax,
    N1=low_num,
    N2=high_num,
    nbr_points=7,
    scale=5,
    method=lambda N4s, weights: mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_extended_return_values(
        N4s=N4s,
        weights=weights,
        func_interpolation_rotation=mechinterfabric.rotation.average_Manton2004,
    )[
        0
    ],
    origin_y=0,
    origin_z=0,
    plot_func_key=plot_func_key,
)


upper = 3
lower = 0
offset = 1
limits = [
    (lower - offset, upper + offset),
    (-0.5 - offset, 0.5 + offset),
    (-0.5 - offset, 0.5 + offset),
]
ax.set_xlim(limits[0])
ax.set_ylim(limits[1])
ax.set_zlim(limits[2])

# Axes labels
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Homogeneous axes
bbox_min = np.min(limits)
bbox_max = np.max(limits)
ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

key = "cubic"

ax.set_title(key)
directory = os.path.join("output", "s040")
os.makedirs(directory, exist_ok=True)
path_picture = os.path.join(directory, key + ".png")
plt.savefig(path_picture)
