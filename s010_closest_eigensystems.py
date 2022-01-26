import numpy as np
import mechinterfabric
import os
import matplotlib.pyplot as plt
import mechkit
import os
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s007")
os.makedirs(directory, exist_ok=True)

np.random.seed(seed=100)

#########################################################

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


biblio = mechkit.fabric_tensors.Basic().N2

pairs = {
    "special case": (
        np.diag([0.95, 0.05, 0]),
        np.diag([0.0, 0.95, 0.05]),
    ),
}

for key, (N2_1, N2_2) in pairs.items():
    print("###########")
    print(key)

    rotations = 