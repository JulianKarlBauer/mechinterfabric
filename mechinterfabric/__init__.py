import numpy as np
import mechkit

converter = mechkit.notation.Converter()


def get_deviator(mandel):
    tensor = converter.to_tensor(mandel)
    deviator = mechkit.operators.dev(tensor)
    return converter.to_mandel6(deviator)


def contract_FOT4_to_FOT2(mandel):
    I2 = np.eye(3)
    tensor_FOT2 = np.tensordot(converter.to_tensor(mandel), I2)
    return tensor_FOT2


class FourthOrderFabricAnalyser:
    def __init__(self):
        return None

    def analyse(self, FOT4):
        # Contract

        FOT2 = contract_FOT4_to_FOT2(FOT4)
        print(FOT2)

        # Identify symmetry FOT2
        # Get eigensystem candidates

        FOT4_dev = get_deviator(FOT4)
        print(FOT4_dev)

        # Get parameters
        result = {"d0": 0.0, "rotation_Q": np.eye(3)}
        return result
