import os

import mechkit
import numpy as np
import pandas as pd


np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("export")
os.makedirs(directory, exist_ok=True)

converter = mechkit.notation.ExplicitConverter()

#########################################################
# Read

df = pd.read_csv(os.path.join(directory, "df.csv"))
new = pd.read_csv(os.path.join(directory, "new.csv"))

from export import df_N4s

df_N4s = df_N4s.data

from export import new_N4s_interpolate_N4_decomp_unique_rotation as tmp

new_N4s = tmp.data

# If you have any problems, make sure to restart the Kernel,
# as import statements create side-effects

print("Initial data")
print(df)

print("Points to be interpolated on")
print(new)

print()
print("Initial N4s of shape")
print(df_N4s.shape)

print("Interpolated N4s of shape")
print(new_N4s.shape)
