import numpy as np
import mechinterfabric
import os
import mechkit
from mayavi import mlab

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

pairs = {
    "Random: many vs few": (
        from_vectors(random_in_between((8, 3))),
        from_vectors(random_in_between((2, 3))),
    ),
}

for key, (N4_1, N4_2) in pairs.items():
    print("###########")
    print(key)

    fig = mlab.figure(figure="ODF", size=(900, 900))

    mechinterfabric.visualization.plot_stepwise_interpolation_N4_along_x_mayavi(
        fig=fig,
        N1=N4_1,
        N2=N4_2,
        nbr_points=5,
        nbr_vectors=50,
        scale=2,
        method=mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation,
        origin_y=0,
        origin_z=0,
    )
mlab.orientation_axes()
mlab.savefig(filename=os.path.join(directory, "image.png"))
mlab.show()

