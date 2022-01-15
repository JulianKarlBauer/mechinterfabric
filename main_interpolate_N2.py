import numpy as np
import mechinterfabric


N1 = np.diag([1, 0, 0])
N2 = np.diag([0.3, 0.3, 0.3])

bunch = np.array([N1, N2])

av = mechinterfabric.interpolation.average_N2(
    bunch, weights=np.ones(len(bunch)) / len(bunch)
)
