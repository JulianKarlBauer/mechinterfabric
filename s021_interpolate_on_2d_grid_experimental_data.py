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


df = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_01_31_1124_N2.csv"),
    header=0,
    sep=",",
)
df.columns = df.columns.str.strip()


def N2_from_row(row):
    return np.array(
        [
            [row["n11"], row["n12"], row["n13"]],
            [row["n12"], row["n22"], row["n23"]],
            [row["n13"], row["n23"], row["n33"]],
        ]
    )


df["N2"] = df.apply(N2_from_row, axis=1)
