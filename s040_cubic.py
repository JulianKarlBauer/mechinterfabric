import vofotensors
from pprint import pprint
from vofotensors.abc import d1
import sympy as sp
import numpy as np
import symbolic as sb
from symbolic.numbers import z, one
from symbolic.abc import alpha

N4 = vofotensors.fabric_tensors.N4s_parametric['cubic']['d1']

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

substitutions = ['-1/15', '2/45']

for expr in substitutions:
    print()
    print('d1=', expr)
    pprint(N4.subs({d1: sp.sympify(expr)}))

w = sp.symbols('w', real=True)

low = N4.subs({d1: sp.sympify(substitutions[0])})
high = N4.subs({d1: sp.sympify(substitutions[1])})

combined = w*low + (sp.S(1)-w) * high

#########################
# Rotate

Q_around_z = np.array(
    (
        (sp.cos(alpha), -sp.sin(alpha), z),
        (sp.sin(alpha), sp.cos(alpha), z),
        (z, z, one),
    ),
)

N4_rotated = sp.Matrix(sp.trigsimp(sb.actively_rotate_mandel(mandel=N4, Q=Q_around_z)))

pprint(N4_rotated)

I2 = sb.get_I2()

N2 = sb.mandel(np.tensordot(sb.tensorr(N4), sb.to_numpy(sb.get_I2())))



