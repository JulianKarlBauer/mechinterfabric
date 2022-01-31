import numpy as np
import mechkit
import pandas as pd
import itertools
import os

np.random.seed(seed=100)
np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s011")
os.makedirs(directory, exist_ok=True)

converter = mechkit.notation.Converter()

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

#########################################################
# N4

df = pd.read_csv(
    os.path.join("data", "juliane_blarr_mail_2022_01_31_1124_N4.csv"),
    header=0,
    sep=",",
)
df.columns = df.columns.str.strip()


def N4_from_row(row):
    # N4 is completely index-symmetric
    N4 = np.zeros((3, 3, 3, 3), dtype=np.float64)

    columns = [
        "3333",
        "3332",
        "3322",
        "3222",
        "2222",
        "3331",
        "3321",
        "3221",
        "2221",
        "3311",
        "3211",
        "2211",
        "3111",
        "2111",
        "1111",
    ]

    for column_key in columns:
        permutations = itertools.permutations([int(item) - 1 for item in column_key])
        for perm in permutations:
            N4[perm] = row[column_key]
    return N4


df["N4"] = df.apply(N4_from_row, axis=1)
