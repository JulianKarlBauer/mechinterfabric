import numpy as np

import mechinterfabric

for la1 in np.linspace(1 / 3, 1, 5):
    for la2 in np.linspace((1 - la1) / 2, min(la1, 1 - la1), 5):
        alpha1, alpha3 = mechinterfabric.utils.to_alpha1_alpha3_to(la1=la1, la2=la2)
        la1_back, la2_back = mechinterfabric.utils.to_lambda1_lambda2(alpha1, alpha3)
        print(la1, la2, la1_back, la2_back)
        print(alpha1, alpha3)
        assert np.allclose([la1, la2], [la1_back, la2_back])
