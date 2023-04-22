import os

import matplotlib.pyplot as plt
import mechkit
import numpy as np

import mechinterfabric.visualization_matplotlib

np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s008")
os.makedirs(directory, exist_ok=True)

np.random.seed(seed=100)

#########################################################

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


def random_in_between(shape, lower=-1, upper=1):
    return (upper - lower) * np.random.rand(*shape) + lower


from_vectors = mechkit.fabric_tensors.first_kind_discrete

biblio = mechkit.fabric_tensors.Basic().N4

pairs = {
    "Random: many vs many": (
        from_vectors(random_in_between((8, 3))),
        from_vectors(random_in_between((8, 3))),
    ),
    "Random: many vs few": (
        from_vectors(random_in_between((8, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
    "Random: 1 vs 1": (
        from_vectors(random_in_between((1, 3))),
        from_vectors(random_in_between((1, 3))),
    ),
    "Random: 2 vs 2": (
        from_vectors(random_in_between((2, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
    "Random: 3 vs 3": (
        from_vectors(random_in_between((2, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
    "piso xy vs piso xz": (
        con.to_tensor(biblio["planar_iso_xy"]),
        con.to_tensor(biblio["planar_iso_xz"]),
    ),
    "piso xy vs piso yz": (
        con.to_tensor(biblio["planar_iso_xy"]),
        con.to_tensor(biblio["planar_iso_yz"]),
    ),
    "piso xz vs piso yz": (
        con.to_tensor(biblio["planar_iso_xz"]),
        con.to_tensor(biblio["planar_iso_yz"]),
    ),
    "iso vs piso xy": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["planar_iso_xy"]),
    ),
    "iso vs ud x": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["ud_x"]),
    ),
}

for key, (N4_1, N4_2) in pairs.items():
    print("###########")
    print(key)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    plot_func_key = "cos_projection_fodf"

    mechinterfabric.visualization_matplotlib.plot_stepwise_interpolation_N4_along_x(
        ax=ax,
        N1=N4_1,
        N2=N4_2,
        nbr_points=5,
        scale=3,
        method=mechinterfabric.interpolation.interpolate_N4_decomp,
        origin_y=0,
        origin_z=0,
        plot_func_key=plot_func_key,
    )

    mechinterfabric.visualization_matplotlib.plot_stepwise_interpolation_N4_along_x(
        ax=ax,
        N1=N4_1,
        N2=N4_2,
        nbr_points=5,
        scale=3,
        method=mechinterfabric.interpolation.interpolate_N4_naive,
        origin_y=0,
        origin_z=1,
        plot_func_key=plot_func_key,
    )

    upper = 2
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

    ax.set_title(key)
    path_picture = os.path.join(directory, key + ".png")
    plt.savefig(path_picture)
