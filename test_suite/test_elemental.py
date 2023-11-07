from src.elementary_functions import sin, cos, tan, exp
from src.dual import DualNumber as DN
import numpy as np

class Test_elemental_functions:
    
    def test_sin(self):
        dual1 = DN(1, 2)
        num_input = np.pi / 2

        sin_of_dual = sin(dual1)
        assert sin_of_dual.real == np.sin(1)
        assert sin_of_dual.dual == 2 * np.cos(1)

        sin_of_num = sin(num_input)
        assert sin_of_num.real == 1
        assert sin_of_num.dual == 1
    
    def test_cos(self):
        dual1 = DN(1, 2)
        num_input = np.pi

        cos_of_dual = cos(dual1)
        assert cos_of_dual.real == np.cos(1)
        assert cos_of_dual.dual == -np.sin(1) * 2

        cos_of_num = cos(num_input)
        assert cos_of_num.real == -1
        assert cos_of_num.dual == 1
    
    def test_tan(self):
        dual1 = DN(1, 2)
        num_input = 5.5

        tan_of_dual = tan(dual1)
        assert tan_of_dual.real == np.tan(1)
        assert tan_of_dual.dual == 2 / (np.cos(1) ** 2)

        tan_of_num = tan(num_input)
        assert tan_of_num.real == np.tan(5.5)
        assert tan_of_num.dual == 1

    def test_exp(self):
        # should this be for any exponent?
        dual1 = DN(2, 3)
        num_input = 6

        exp_of_dual = exp(dual1)
        assert exp_of_dual.real == np.e ** 2
        assert exp_of_dual.dual == 3 * np.e ** 2

        exp_of_num = exp(num_input)
        assert exp_of_num.real == np.e ** 6
        assert exp_of_num.dual == 1