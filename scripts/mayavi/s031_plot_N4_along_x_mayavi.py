import os

import mechkit
import numpy as np
from mayavi import mlab
from scipy.spatial.transform import Rotation

import mechinterfabric

np.set_printoptions(linewidth=100000)

directory = os.path.join("output", "s031")
os.makedirs(directory, exist_ok=True)

np.random.seed(seed=100)

#########################################################

converter = mechkit.notation.ExplicitConverter()
con = mechkit.notation.Converter()


def random_in_between(shape, lower=-1, upper=1):
    return (upper - lower) * np.random.rand(*shape) + lower


from_vectors = mechkit.fabric_tensors.first_kind_discrete

biblio = mechkit.fabric_tensors.Basic().N4

small_rotation_around_z = Rotation.from_rotvec(-1e-2 * np.array([0, 0, 1])).as_matrix()

pairs = {
    # "Random: many vs few": (
    #     from_vectors(random_in_between((1, 3))),
    #     from_vectors(random_in_between((1, 3))),
    # ),
    "ud z vs ud x": (
        con.to_tensor(biblio["ud_z"]),
        np.einsum(
            "io,jm,kn,lp, omnp->ijkl",
            small_rotation_around_z,
            small_rotation_around_z,
            small_rotation_around_z,
            small_rotation_around_z,
            con.to_tensor(biblio["ud_x"]),
        ),
    ),
}

for key, (N4_1, N4_2) in pairs.items():
    print("###########")
    print(key)

    fig = mlab.figure(
        figure="ODF", size=(1400, 900), bgcolor=(1, 1, 1), fgcolor=(0.0, 0.0, 0.0)
    )

    scale = 5
    offest = 1.5

    upper = mechinterfabric.visualization.plot_stepwise_interpolation_N4_along_x_mayavi(
        fig=fig,
        N1=N4_1,
        N2=N4_2,
        nbr_points=5,
        nbr_vectors=100,
        scale=scale,
        method=mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation,
        origin_y=0,
        origin_z=0,
    )

    lower = mechinterfabric.visualization.plot_stepwise_interpolation_N4_along_x_mayavi(
        fig=fig,
        N1=N4_1,
        N2=N4_2,
        nbr_points=5,
        nbr_vectors=100,
        scale=scale,
        method=mechinterfabric.interpolation.interpolate_N4_naive,
        origin_y=0,
        origin_z=offest,
    )

kwargs = dict(line_width=5, color=(0, 0, 0))
mlab.outline(upper[0], **kwargs)
mlab.outline(upper[-1], **kwargs)
mlab.outline(lower[0], **kwargs)
mlab.outline(lower[-1], **kwargs)

if True:
    view = mlab.view()
    (azimuth, elevation, distance, focalpoint) = view
    mlab.view(*(-90, 90, distance, focalpoint))

mlab.gcf().scene.parallel_projection = True

mlab.orientation_axes()
mlab.savefig(filename=os.path.join(directory, "image.png"))

mlab.show()
