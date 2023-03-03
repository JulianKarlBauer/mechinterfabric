import mechkit
import numpy as np
import scipy

import mechinterfabric


converter = mechkit.notation.Converter()


def rotate_fot4_randomly(fot4):
    return converter.to_mandel6(
        mechinterfabric.utils.rotate(converter.to_tensor(fot4), Q=get_random_rotation())
    )


def get_random_rotation():
    angle = 2 * np.pi * np.random.rand(1)
    rotation_vector = np.array(np.random.rand(3))

    rotation = scipy.spatial.transform.Rotation.from_rotvec(
        angle * rotation_vector, degrees=False
    )
    return rotation.as_matrix()
