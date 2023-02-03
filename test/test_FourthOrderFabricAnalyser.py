import mechinterfabric
import numpy as np


class TestAnalyser:
    def test_init(self):
        analyser = mechinterfabric.FourthOrderFabricAnalyser()

    def test_analyse(self):
        analyser = mechinterfabric.FourthOrderFabricAnalyser()
        assert np.allclose(analyser.analyse(np.eye(6)).result["rotation_Q"], np.eye(3))
