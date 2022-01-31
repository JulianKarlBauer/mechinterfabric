import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import mechkit
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem
import os
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s011")
os.makedirs(directory, exist_ok=True)

#########################################################
# N2

df_N2 = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_01_31_1124_N2.csv"),
    header=0,
    sep=",",
)
df_N2.columns = df_N2.columns.str.strip()


def N2_from_row(row):
    return np.array(
        [
            [row["n11"], row["n12"], row["n13"]],
            [row["n12"], row["n22"], row["n23"]],
            [row["n13"], row["n23"], row["n33"]],
        ]
    )


df_N2["N2"] = df_N2.apply(N2_from_row, axis=1)
