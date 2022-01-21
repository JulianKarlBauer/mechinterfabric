import numpy as np
import mechinterfabric
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import mechkit

directory = os.path.join("output", "s002")
os.makedirs(directory, exist_ok=True)

con = mechkit.notation.Converter()

biblio = mechkit.fabric_tensors.Basic().N2

pairs = {
    "special case": (
        np.diag([0.95, 0.05, 0]),
        np.diag([0.0, 0.95, 0.05]),
    ),
    # "ud x vs ud y": (
    #     con.to_tensor(biblio["ud_x"]),
    #     con.to_tensor(biblio["ud_y"]),
    # ),
    # "ud x vs ud z": (
    #     con.to_tensor(biblio["ud_x"]),
    #     con.to_tensor(biblio["ud_z"]),
    # ),
    # "ud y vs ud z": (
    #     con.to_tensor(biblio["ud_y"]),
    #     con.to_tensor(biblio["ud_z"]),
    # ),
    "iso vs ud x": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["ud_x"]),
    ),
    "iso vs ud y": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["ud_y"]),
    ),
    "iso vs ud z": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["ud_z"]),
    ),
    "iso vs piso xy": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["planar_iso_xy"]),
    ),
    "iso vs piso xz": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["planar_iso_xz"]),
    ),
    "iso vs piso yz": (
        con.to_tensor(biblio["iso"]),
        con.to_tensor(biblio["planar_iso_yz"]),
    ),
}

for key, (N1, N2) in pairs.items():
    print("###########")
    print(key)

    bunch = np.array([N1, N2])

    av, av_in_eigen, av_rotation = mechinterfabric.interpolation.interpolate_N2_decomp(
        bunch, weights=np.ones(len(bunch)) / len(bunch)
    )

    # Plot

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Axes labels
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    mechinterfabric.visualization.plot_stepwise_interpolation_along_x(
        ax, N1, N2, nbr_points=5, scale=4
    )

    bbox_min = -2
    bbox_max = 2
    ax.auto_scale_xyz([bbox_min, bbox_max], [bbox_min, bbox_max], [bbox_min, bbox_max])

    ax.set_title(key)
    path_picture = os.path.join(directory, key + ".png")
    plt.savefig(path_picture)
